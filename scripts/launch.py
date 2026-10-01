"""Bring up a chosen local LLM stack: model server, front door, and web UI.

The pieces are independent, and which ones you want depends on the job:

    launch.py                                  # UI + API on loopback, no token
    launch.py --model 31b --think on           # bigger model, reasoning on
    launch.py --no-ui                          # API only, for scripts
    launch.py --expose lan --auth              # reachable from a phone, token required

The model server itself always binds to loopback. Anything that needs to be
reachable from elsewhere goes through the front door instead, so exposing the
stack and authenticating it are the same decision rather than two that can drift
apart.
"""

from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from resolve_model import load as load_models  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = ROOT / "web"


def port_busy(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0


def preflight(ports: dict[int, str]) -> bool:
    """Refuse to start if a port is taken, naming which and why it matters.

    Without this the readiness poll below is actively dangerous: an unrelated
    server already listening on the model port answers it, so the launcher
    reports success for a model it never loaded, and every later request goes to
    whatever else is running there.
    """
    busy = {p: role for p, role in ports.items() if port_busy(p)}
    if not busy:
        return True
    for port, role in busy.items():
        print(f"error: port {port} ({role}) is already in use.", file=sys.stderr)
    print(
        "\nSomething is already running -- probably an earlier stack.\n"
        "Stop it and start again in one step:\n"
        "    make restart MODEL=...\n"
        "or stop it on its own with `make down`, or pick other ports with\n"
        "--port / --ui-port.",
        file=sys.stderr,
    )
    return False


def check_runtime(key: str, models: dict) -> bool:
    """Reject a model mlx_lm.server cannot load, before anything starts.

    mlx_lm.server loads the weights lazily, on the first completion request.
    An unsupported architecture therefore starts cleanly, answers /v1/models,
    and only fails when someone finally types something -- as a bare HTTP 404
    that names no cause. Catching it here turns a confusing runtime failure into
    a startup error that says what to do.
    """
    spec = models.get(key)
    if spec is None:
        return True  # A raw HF id; we have no registry entry to check it against.
    runtime = spec.get("runtime", "mlx-lm")
    if runtime == "mlx-lm":
        return True
    servable = [k for k, v in models.items() if v.get("runtime", "mlx-lm") == "mlx-lm"]
    print(
        f"error: {key} ({spec['id']}) has model_type "
        f"{spec.get('model_type', 'unknown')!r}, which needs {runtime}.\n"
        f"mlx_lm.server cannot load it, so `make up` would fail on the first\n"
        f"message with an unexplained 404.\n\n"
        f"Servable models: {', '.join(servable)}",
        file=sys.stderr,
    )
    return False


def hf_cached(repo_id: str) -> bool:
    """True only when the weights are fully present.

    The cache directory appears as soon as a download starts, so checking that
    it exists treats a half-finished model as ready: --pull never would let the
    launcher through, and the load would fail much later with a confusing error.
    Hugging Face marks partial blobs with a .incomplete suffix, so look for those.
    """
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    root = cache / f"models--{repo_id.replace('/', '--')}"
    if not root.is_dir():
        return False
    if any(root.glob("blobs/*.incomplete")):
        return False
    return any(root.glob("snapshots/*/*.safetensors"))


def human_gb(models: dict, key: str) -> str:
    spec = models.get(key)
    return f"~{spec['mem_gb']:.1f} GB" if spec else "unknown size"


def ensure_model(repo_id: str, key: str, models: dict, policy: str) -> bool:
    if hf_cached(repo_id):
        return True
    if policy == "never":
        print(f"error: {repo_id} is not downloaded. Run `make pull MODEL={key}`.", file=sys.stderr)
        return False
    if policy == "ask":
        size = human_gb(models, key)
        print(f"\n{repo_id} is not downloaded yet ({size}).")
        # Large transfers are worth a deliberate yes rather than a silent start.
        try:
            answer = input("Download it now? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in {"y", "yes"}:
            print("Aborted.")
            return False
    print(f"Downloading {repo_id} ...")
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "pull.py"), key]).returncode == 0


def wait_until_ready(url: str, proc: subprocess.Popen, timeout: float = 300) -> bool:
    """Poll until the model server answers, or it dies trying."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"\nerror: model server exited with code {proc.returncode}", file=sys.stderr)
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            pass
        time.sleep(0.5)
        print(".", end="", flush=True)
    print("\nerror: model server did not become ready in time", file=sys.stderr)
    return False


def main() -> int:
    models = load_models()

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="12b", help=f"model key or HF id ({', '.join(models)})")
    ap.add_argument("--think", choices=["on", "off"], default="off", help="Gemma thinking mode")
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--ui", dest="ui", action="store_true", default=True)
    ap.add_argument("--no-ui", dest="ui", action="store_false", help="API only, no web page")
    ap.add_argument("--auth", dest="auth", action="store_true", default=False)
    ap.add_argument("--no-auth", dest="auth", action="store_false")
    ap.add_argument(
        "--expose",
        choices=["loopback", "lan"],
        default="loopback",
        help="loopback: this machine only. lan: other devices, which forces --auth.",
    )
    ap.add_argument("--pull", choices=["ask", "auto", "never"], default="ask")
    ap.add_argument("--port", type=int, default=8080, help="model server (always loopback)")
    ap.add_argument("--ui-port", type=int, default=8443, help="front door")
    args = ap.parse_args()

    # Reaching the stack from another device without a credential is not a
    # trade-off worth offering, so exposure implies authentication.
    if args.expose == "lan" and not args.auth:
        print("note: --expose lan requires authentication; enabling --auth.")
        args.auth = True

    # A token file every account on the machine can read is barely a secret, and
    # the default umask produces exactly that. Warn rather than fail: the file is
    # the user's, and the fix is one command.
    if args.auth:
        env_file = ROOT / ".env"
        if env_file.is_file() and env_file.stat().st_mode & 0o077:
            print(
                f"warning: {env_file.name} is readable by other accounts "
                f"({oct(env_file.stat().st_mode & 0o777)}).\n"
                f"         Fix with: chmod 600 .env",
                file=sys.stderr,
            )

    if not check_runtime(args.model, models):
        return 1

    wanted = {args.port: "model server"}
    if args.ui or args.auth or args.expose == "lan":
        wanted[args.ui_port] = "front door"
    if not preflight(wanted):
        return 1

    repo_id = models[args.model]["id"] if args.model in models else args.model
    if not ensure_model(repo_id, args.model, models, args.pull):
        return 1

    think = "true" if args.think == "on" else "false"
    procs: list[subprocess.Popen] = []

    server_cmd = [
        "mlx_lm.server",
        "--model", repo_id,
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--max-tokens", str(args.max_tokens),
        "--chat-template-args", f'{{"enable_thinking":{think}}}',
    ]  # fmt: skip
    if shutil.which(server_cmd[0]) is None:
        print("error: mlx_lm.server not found. Run `make setup`.", file=sys.stderr)
        return 1

    print(f"Model      {repo_id}")
    print(f"Thinking   {args.think}")
    print("Loading", end="", flush=True)

    procs.append(subprocess.Popen(server_cmd))
    try:
        if not wait_until_ready(f"http://127.0.0.1:{args.port}/v1/models", procs[0]):
            return 1
        print(" ready")

        # The front door only earns its place when something it provides is
        # wanted: the page, a credential, or reachability from off-box.
        need_front = args.ui or args.auth or args.expose == "lan"
        if need_front:
            bind = "0.0.0.0" if args.expose == "lan" else "127.0.0.1"
            front = [
                sys.executable, str(ROOT / "scripts" / "auth_proxy.py"),
                "--listen", f"{bind}:{args.ui_port}",
                "--upstream", f"127.0.0.1:{args.port}",
            ]  # fmt: skip
            if args.ui:
                servable = [
                    v["id"] for v in models.values() if v.get("runtime", "mlx-lm") == "mlx-lm"
                ]
                front += [
                    "--web-root", str(WEB_ROOT),
                    "--active-model", repo_id,
                    "--servable", ",".join(servable),
                ]  # fmt: skip
            if not args.auth:
                front.append("--no-auth")
            procs.append(subprocess.Popen(front))
            time.sleep(1)
            if procs[-1].poll() is not None:
                return 1

            where = "0.0.0.0" if args.expose == "lan" else "127.0.0.1"
            print()
            if args.ui:
                print(f"UI         http://{where}:{args.ui_port}/")
            print(f"API        http://{where}:{args.ui_port}/v1")
            print(f"Auth       {'bearer token (MLX_API_TOKEN)' if args.auth else 'none'}")
        else:
            print(f"\nAPI        http://127.0.0.1:{args.port}/v1 (loopback, no auth)")

        print("\nCtrl-C to stop.")
        while all(p.poll() is None for p in procs):
            time.sleep(0.5)
        print("A component exited on its own.")
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping ...")
        for p in reversed(procs):
            if p.poll() is None:
                p.terminate()
        for p in reversed(procs):
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        print("Stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
