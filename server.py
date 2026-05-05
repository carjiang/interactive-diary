"""Persistent in-container server so the NER model loads once and stays cached."""
from __future__ import annotations

import json
import os
import tempfile
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self._respond(200, b"ok")
        else:
            self._respond(404, b"not found")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        diary_entry = body["diary_entry"]
        user_id     = body["user_id"]
        session_id  = body["session_id"]
        top_k       = int(body["top_k"])

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            output_path = f.name

        try:
            if self.path == "/rag":
                from interface.io_rag import generate_rag_response
                generate_rag_response(diary_entry, user_id, session_id, top_k, output_path)
            elif self.path == "/gpt":
                from interface.io_rag import generate_gpt_response
                generate_gpt_response(diary_entry, user_id, session_id, top_k, output_path)
            else:
                self._respond(404, b"unknown endpoint")
                return

            with open(output_path) as f:
                text = f.read()
        finally:
            os.unlink(output_path)

        self._respond(200, text.encode())

    def _respond(self, code: int, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # silence per-request logs


class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    port = int(os.environ.get("SERVER_PORT", 8765))
    server = _ThreadedHTTPServer(("0.0.0.0", port), _Handler)
    print(f"Server ready on port {port}", flush=True)
    server.serve_forever()
