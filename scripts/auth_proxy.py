"""Bearer-token reverse proxy in front of mlx_lm.server.

mlx_lm.server has no authentication of any kind. Its only protection is the
default bind address of 127.0.0.1, which keeps it unreachable from the network.
The moment that becomes --host 0.0.0.0, the model is open to everyone on the
same LAN with no credential at all.

This proxy restores the missing control. Run mlx_lm.server on loopback as usual,
point this at it, and expose *this* instead:

    mlx_lm.server --model ... --port 8080          # stays on 127.0.0.1
    MLX_API_TOKEN=... auth_proxy.py --listen 0.0.0.0:8443 --upstream 127.0.0.1:8080

Standard library only, deliberately: an authentication component is the last
place to add dependencies.

Scope and limits, so this is not mistaken for more than it is:

  * The token travels in plaintext. This is for a trusted LAN, not the internet.
    For anything beyond that, terminate TLS in front of it.
  * There is one shared token, not per-user accounts. It answers "is this
    caller allowed", not "who is this caller".
  * There is no rate limiting.
"""

from __future__ import annotations

import argparse
import hmac
import os
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Headers that describe a single hop and must not be copied between connections.
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}

MAX_BODY = 32 * 1024 * 1024  # Refuse absurd uploads rather than buffering them.


class AuthProxy(BaseHTTPRequestHandler):
    upstream = "127.0.0.1:8080"
    token = ""
    protocol_version = "HTTP/1.1"
    server_version = "mlx-auth-proxy"
    sys_version = ""

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(f"{self.address_string()} {fmt % args}\n")

    def _authorized(self) -> bool:
        header = self.headers.get("Authorization", "")
        scheme, _, presented = header.partition(" ")
        if scheme.lower() == "bearer" and presented:
            candidate = presented.strip()
        else:
            # Anthropic-style clients send the credential this way instead.
            candidate = (self.headers.get("x-api-key") or "").strip()
        if not candidate:
            return False
        # Constant time: a plain == leaks the shared prefix through timing.
        return hmac.compare_digest(candidate, self.token)

    def _deny(self) -> None:
        self.log_message("401 %s %s", self.command, self.path)
        body = b'{"error":{"message":"missing or invalid credential","type":"unauthorized"}}'
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("WWW-Authenticate", 'Bearer realm="mlx"')
        self.end_headers()
        self.wfile.write(body)

    def _forward(self) -> None:
        if not self._authorized():
            self._deny()
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self.send_error(413, "request body too large")
            return
        body = self.rfile.read(length) if length else None

        headers = {
            k: v
            for k, v in self.headers.items()
            # Drop the client's credential; upstream neither needs nor checks it.
            if k.lower() not in HOP_BY_HOP | {"host", "authorization", "x-api-key"}
        }

        req = urllib.request.Request(
            f"http://{self.upstream}{self.path}",
            data=body,
            headers=headers,
            method=self.command,
        )

        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                self.send_response(resp.status)
                for key, value in resp.headers.items():
                    if key.lower() not in HOP_BY_HOP | {"content-length"}:
                        self.send_header(key, value)
                # Length is unknown up front for streamed completions, so relay
                # with chunked encoding and flush each chunk as it arrives --
                # buffering here would defeat token-by-token streaming.
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                while chunk := resp.read(8192):
                    self.wfile.write(f"{len(chunk):X}\r\n".encode())
                    self.wfile.write(chunk)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            self.send_response(exc.code)
            self.send_header("Content-Type", exc.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            self.log_message("upstream unreachable: %s", exc)
            self.send_error(502, "upstream unreachable")

    do_GET = _forward
    do_POST = _forward
    do_DELETE = _forward


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--listen", default="127.0.0.1:8443", metavar="HOST:PORT")
    ap.add_argument("--upstream", default="127.0.0.1:8080", metavar="HOST:PORT")
    args = ap.parse_args()

    token = os.environ.get("MLX_API_TOKEN", "").strip()
    if not token:
        print(
            "error: MLX_API_TOKEN is not set.\n"
            "Refusing to start: an unauthenticated proxy is worse than no proxy,\n"
            "because it looks protected. Generate one with:\n"
            "    python3 -c 'import secrets; print(secrets.token_urlsafe(32))'",
            file=sys.stderr,
        )
        return 1
    if len(token) < 16:
        print("error: MLX_API_TOKEN is shorter than 16 characters.", file=sys.stderr)
        return 1

    host, _, port = args.listen.rpartition(":")
    AuthProxy.upstream = args.upstream
    AuthProxy.token = token

    server = ThreadingHTTPServer((host, int(port)), AuthProxy)
    print(f"Authenticating proxy on {args.listen} -> {args.upstream}")
    if host not in {"127.0.0.1", "localhost", "::1"}:
        print(f"WARNING: bound to {host}, reachable from the network. Token is sent in plaintext.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
