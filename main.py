#!/usr/bin/env python3
# SSE normalization: upstream SSE may include blank lines or comment lines (starting with ':').
# Per SSE spec, these must be relayed as-is with a trailing blank line to preserve event framing.
# relay_openai_stream now writes non-data lines with an extra newline for spec compliance.
import os
import sys
import json
import time
import threading
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE")
GATEWAY = os.environ.get("OPENAI_BASE_URL", "")
API_KEY = os.environ.get("OPENAI_API_KEY", "")

BRIEF_MODELS = []
BRIEF_PROMPTS = {}

def parse_brief():
    global BRIEF_MODELS, BRIEF_PROMPTS
    if not PROMPT_FILE or not os.path.exists(PROMPT_FILE):
        return
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                schema = doc.get("schema", "")
                if schema == "free-model-aggregator-02/models/v1":
                    BRIEF_MODELS = doc.get("models", [])
                elif schema == "free-model-aggregator-02/prompts/v1":
                    BRIEF_PROMPTS = doc
    except Exception:
        pass

def build_catalog():
    providers = []
    free_models = [m for m in BRIEF_MODELS if m.get("free", False)]
    if free_models:
        models = []
        for idx, m in enumerate(free_models):
            models.append({
                "id": m.get("id", ""),
                "free": True,
                "context_length": m.get("context_length", 0)
            })
        providers.append({
            "id": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "key_env": "OPENROUTER_API_KEY",
            "wire": "openai",
            "models": models
        })
    return {"providers": providers}

parse_brief()
CATALOG = build_catalog()

if "--list" in sys.argv:
    print(json.dumps(CATALOG, separators=(",", ":"), ensure_ascii=False))
    sys.exit(0)

COOLED = set()
COOLED_LOCK = threading.Lock()

def model_key(m):
    return m.get("id", "")

def select_candidates(body):
    needs_tools = bool(body.get("tools"))
    needs_json = False
    rf = body.get("response_format")
    if isinstance(rf, dict):
        t = rf.get("type")
        if t in ("json_object", "json_schema"):
            needs_json = True
    needs_vision = False
    for msg in body.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    pt = part.get("type")
                    if pt in ("image_url", "image"):
                        needs_vision = True
                        break
        elif isinstance(content, str):
            pass
    candidates = []
    for idx, m in enumerate(BRIEF_MODELS):
        if not m.get("free", False):
            continue
        if needs_tools and not m.get("tools", False):
            continue
        if needs_json and not m.get("json", False):
            continue
        if needs_vision and not m.get("vision", False):
            continue
        candidates.append((m, idx))
    candidates.sort(key=lambda x: (-x[0].get("rating_milli", 0), x[1]))
    with COOLED_LOCK:
        cooled = set(COOLED)
    non_cooled = [(m, i) for m, i in candidates if model_key(m) not in cooled]
    cooled_list = [(m, i) for m, i in candidates if model_key(m) in cooled]
    ordered = non_cooled + cooled_list
    return [m for m, _ in ordered[:6]]

def parse_error_code(resp_body_bytes):
    try:
        err = json.loads(resp_body_bytes.decode("utf-8"))
        e = err.get("error", {})
        code = e.get("code") or e.get("type")
        if isinstance(code, str):
            return code
    except Exception:
        pass
    return ""

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/v1/models":
            self.handle_models()
        else:
            self.send_error_json(404, "not found", "invalid_request_error", "not_found")

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self.handle_chat(False)
        elif self.path == "/v1/messages":
            self.handle_chat(True)
        else:
            self.send_error_json(404, "not found", "invalid_request_error", "not_found")

    def handle_models(self):
        api_key = self.headers.get("x-api-key", "")
        free_models = [m for m in BRIEF_MODELS if m.get("free", False)]
        if api_key:
            data = []
            for m in free_models:
                data.append({
                    "type": "model",
                    "id": m.get("id", ""),
                    "display_name": m.get("display_name", ""),
                    "created_at": "2024-01-01T00:00:00Z"
                })
            resp = {
                "data": data,
                "has_more": False,
                "first_id": data[0]["id"] if data else None,
                "last_id": data[-1]["id"] if data else None
            }
        else:
            data = []
            for m in free_models:
                data.append({
                    "id": m.get("id", ""),
                    "object": "model",
                    "created": 0,
                    "owned_by": "openrouter"
                })
            resp = {"object": "list", "data": data}
        self.send_json(200, resp)

    def handle_chat(self, is_anthropic):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body_bytes = self.rfile.read(length)
            body = json.loads(body_bytes.decode("utf-8"))
        except Exception:
            self.send_error_json(400, "invalid request", "invalid_request_error", "invalid_request")
            return

        trail = []
        candidates = select_candidates(body)

        for model in candidates:
            dialled_id = model.get("id", "")
            status, resp_body, resp_headers = self.call_upstream(body, dialled_id, is_anthropic)
            if status == 200:
                if is_anthropic and body.get("stream"):
                    self.relay_anthropic_stream(resp_body, dialled_id, trail)
                elif is_anthropic:
                    self.send_anthropic_response(resp_body, dialled_id, trail)
                elif body.get("stream"):
                    self.relay_openai_stream(resp_body, dialled_id, trail)
                else:
                    self.send_openai_response(resp_body, dialled_id, trail)
                return
            else:
                code = parse_error_code(resp_body)
                trail.append({"status": status, "code": code, "model": dialled_id})
                if len(trail) > 8:
                    trail = trail[-8:]
                if status == 429:
                    with COOLED_LOCK:
                        COOLED.add(dialled_id)

        if trail:
            last = trail[-1]
            self.send_response(last["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("x-makefun-upstream-failures", json.dumps(trail, separators=(",", ":"), ensure_ascii=False))
            self.end_headers()
            self.wfile.write(resp_body)
        else:
            self.send_error_json(500, "no models available", "api_error", "internal")

    def call_upstream(self, body, model_id, is_anthropic):
        if is_anthropic:
            upstream_body = self.translate_anthropic_to_openai(body, model_id)
        else:
            upstream_body = self.prepare_openai_body(body, model_id)
        data = json.dumps(upstream_body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            GATEWAY + "/chat/completions",
            data=data,
            headers={
                "Authorization": "Bearer " + API_KEY,
                "Content-Type": "application/json"
            },
            method="POST"
        )
        stream = bool(upstream_body.get("stream"))
        try:
            resp = urllib.request.urlopen(req, timeout=25)
            if stream:
                return resp.status, resp, resp.headers
            with resp:
                return resp.status, resp.read(), resp.headers
        except urllib.error.HTTPError as e:
            return e.code, e.read(), e.headers
        except Exception as e:
            return 500, json.dumps({"error": {"message": str(e), "type": "api_error", "code": "internal"}}, separators=(",", ":"), ensure_ascii=False).encode("utf-8"), {}

    def prepare_openai_body(self, body, model_id):
        b = dict(body)
        b["model"] = model_id
        if "max_tokens" not in b and "max_completion_tokens" not in b:
            b["max_tokens"] = 1024
        for key in ("max_tokens", "max_completion_tokens"):
            if key in b and isinstance(b[key], (int, float)) and b[key] > 8192:
                b[key] = 8192
        if b.get("stream"):
            b["stream_options"] = {"include_usage": True}
        return b

    def translate_anthropic_to_openai(self, body, model_id):
        messages = []
        system_content = body.get("system")
        if isinstance(system_content, str):
            messages.append({"role": "system", "content": system_content})
        elif isinstance(system_content, list):
            texts = [block.get("text", "") for block in system_content if block.get("type") == "text"]
            if texts:
                messages.append({"role": "system", "content": "\n".join(texts)})

        for msg in body.get("messages", []):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, str):
                messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                parts = []
                tool_calls = []
                for block in content:
                    btype = block.get("type")
                    if btype == "text":
                        parts.append({"type": "text", "text": block.get("text", "")})
                    elif btype == "image":
                        source = block.get("source", {})
                        if source.get("type") == "base64":
                            mt = source.get("media_type", "image/png")
                            data = source.get("data", "")
                            parts.append({"type": "image_url", "image_url": {"url": "data:" + mt + ";base64," + data}})
                    elif btype == "tool_use" and role == "assistant":
                        tool_calls.append({
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {}), separators=(",", ":"), ensure_ascii=False)
                            }
                        })
                    elif btype == "tool_result" and role == "user":
                        pass
                if role == "assistant":
                    msg_obj = {"role": "assistant"}
                    if len(parts) == 1 and parts[0]["type"] == "text":
                        msg_obj["content"] = parts[0]["text"]
                    else:
                        msg_obj["content"] = [p for p in parts if p["type"] == "text"]
                        msg_obj["content"] = "\n".join([p["text"] for p in parts if p["type"] == "text"])
                    if tool_calls:
                        msg_obj["tool_calls"] = tool_calls
                    messages.append(msg_obj)
                else:
                    if len(parts) == 1 and parts[0]["type"] == "text":
                        messages.append({"role": "user", "content": parts[0]["text"]})
                    elif parts:
                        messages.append({"role": "user", "content": parts})
                    for block in content:
                        if block.get("type") == "tool_result":
                            raw = block.get("content", "")
                            if isinstance(raw, list):
                                raw = "\n".join(b.get("text", "") for b in raw if isinstance(b, dict) and b.get("type") == "text")
                            messages.append({
                                "role": "tool",
                                "tool_call_id": block.get("tool_use_id", ""),
                                "content": raw
                            })

        max_tokens = body.get("max_tokens", 1024)
        if max_tokens > 8192:
            max_tokens = 8192

        tools = []
        for t in body.get("tools", []):
            tools.append({
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or {"type": "object", "properties": {}}
                }
            })

        tool_choice = body.get("tool_choice")
        tc = None
        if isinstance(tool_choice, dict):
            ttype = tool_choice.get("type")
            if ttype == "auto":
                tc = "auto"
            elif ttype == "any":
                tc = "required"
            elif ttype == "tool":
                tc = {"type": "function", "function": {"name": tool_choice.get("name", "")}}
            elif ttype == "none":
                tc = "none"

        upstream = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": body.get("stream", False)
        }
        if body.get("temperature") is not None:
            upstream["temperature"] = body["temperature"]
        if body.get("stop_sequences"):
            upstream["stop"] = body["stop_sequences"]
        if tools:
            upstream["tools"] = tools
        if tc is not None:
            upstream["tool_choice"] = tc
        if upstream["stream"]:
            upstream["stream_options"] = {"include_usage": True}
        return upstream

    def send_success_headers(self, content_type, trail):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        if trail:
            self.send_header("x-makefun-upstream-failures", json.dumps(trail, separators=(",", ":"), ensure_ascii=False))
        self.end_headers()

    def send_openai_response(self, resp_body, model_id, trail):
        self.send_success_headers("application/json", trail)
        try:
            data = json.loads(resp_body.decode("utf-8"))
        except Exception:
            data = {}
        data["model"] = model_id
        out = json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.wfile.write(out)

    def relay_openai_stream(self, resp, model_id, trail):
        self.send_success_headers("text/event-stream", trail)
        self.close_connection = True
        for line in resp:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                payload = line[6:].strip()
                if payload == "[DONE]":
                    self.wfile.write(b"data: [DONE]\n\n")
                    continue
                try:
                    chunk = json.loads(payload)
                    chunk["model"] = model_id
                    out = json.dumps(chunk, separators=(",", ":"), ensure_ascii=False)
                    self.wfile.write(("data: " + out + "\n\n").encode("utf-8"))
                except Exception:
                    self.wfile.write((line.rstrip("\n") + "\n\n").encode("utf-8"))
            else:
                self.wfile.write((line.rstrip("\n") + "\n\n").encode("utf-8"))
            self.wfile.flush()

    def send_anthropic_response(self, resp_body, model_id, trail):
        self.send_success_headers("application/json", trail)
        try:
            data = json.loads(resp_body.decode("utf-8"))
        except Exception:
            data = {}
        msg = data.get("choices", [{}])[0].get("message", {})
        content = msg.get("content")
        tool_calls = msg.get("tool_calls", [])
        usage = data.get("usage", {})

        out_content = []
        if isinstance(content, str) and content:
            out_content.append({"type": "text", "text": content})
        for tc in tool_calls:
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            try:
                parsed = json.loads(args)
            except Exception:
                parsed = {}
            out_content.append({
                "type": "tool_use",
                "id": tc.get("id", ""),
                "name": fn.get("name", ""),
                "input": parsed
            })

        finish_reason = data.get("choices", [{}])[0].get("finish_reason", "stop")
        stop_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "refusal"
        }
        stop_reason = stop_map.get(finish_reason, "end_turn")

        resp = {
            "id": data.get("id", "msg_1"),
            "type": "message",
            "role": "assistant",
            "model": model_id,
            "content": out_content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0)
            }
        }
        out = json.dumps(resp, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.wfile.write(out)

    def relay_anthropic_stream(self, resp, model_id, trail):
        self.send_success_headers("text/event-stream", trail)
        self.close_connection = True
        blocks_started = {}
        last_finish = None
        last_completion_tokens = 0
        started = False

        def emit_start(msg_id):
            self.send_anthropic_event("message_start", {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model_id,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0}
                }
            })

        for line in resp:
            try:
                text = line.decode("utf-8")
            except Exception:
                continue
            if not text.startswith("data: "):
                continue
            payload = text[6:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except Exception:
                continue
            if not started:
                emit_start(chunk.get("id", "msg_1"))
                started = True
            usage = chunk.get("usage")
            if usage:
                last_completion_tokens = usage.get("completion_tokens", last_completion_tokens)
            choices = chunk.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason"):
                last_finish = choice["finish_reason"]
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content:
                if 0 not in blocks_started:
                    self.send_anthropic_event("content_block_start", {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""}
                    })
                    blocks_started[0] = True
                self.send_anthropic_event("content_block_delta", {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": content}
                })
            for tc in delta.get("tool_calls", []):
                idx = tc.get("index", 0) + 1
                fn = tc.get("function", {})
                if idx not in blocks_started:
                    self.send_anthropic_event("content_block_start", {
                        "type": "content_block_start",
                        "index": idx,
                        "content_block": {
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": fn.get("name", ""),
                            "input": {}
                        }
                    })
                    blocks_started[idx] = True
                args_delta = fn.get("arguments", "")
                if args_delta:
                    self.send_anthropic_event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "input_json_delta", "partial_json": args_delta}
                    })

        if not started:
            emit_start("msg_1")
        stop_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "refusal"
        }
        for idx in sorted(blocks_started.keys()):
            self.send_anthropic_event("content_block_stop", {
                "type": "content_block_stop",
                "index": idx
            })
        self.send_anthropic_event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_map.get(last_finish, "end_turn"), "stop_sequence": None},
            "usage": {"output_tokens": last_completion_tokens}
        })
        self.send_anthropic_event("message_stop", {"type": "message_stop"})

    def send_anthropic_event(self, event_type, data):
        out = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
        self.wfile.write(("event: " + event_type + "\ndata: " + out + "\n\n").encode("utf-8"))
        self.wfile.flush()

    def send_json(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        out = json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.wfile.write(out)

    def send_error_json(self, status, message, etype, code):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        err = {"error": {"message": message, "type": etype, "code": code}}
        out = json.dumps(err, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.wfile.write(out)

    def log_message(self, format, *args):
        pass

def register_with_env(port):
    if not ENV_URL:
        return
    payload = {
        "protocol_version": 1,
        "port": port,
        "catalog": CATALOG
    }
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    for _ in range(50):
        try:
            req = urllib.request.Request(
                ENV_URL + "/register",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.2)

def poll_work(server):
    if not ENV_URL:
        return
    while True:
        try:
            with urllib.request.urlopen(ENV_URL + "/work", timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("done"):
                    server.shutdown()
                    break
        except Exception:
            pass
        time.sleep(0.1)

def main():
    parse_brief()
    global CATALOG
    CATALOG = build_catalog()

    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    port = server.server_address[1]

    if ENV_URL:
        register_with_env(port)
        poller = threading.Thread(target=poll_work, args=(server,), daemon=True)
        poller.start()

    server.serve_forever()
    sys.exit(0)

if __name__ == "__main__":
    main()
