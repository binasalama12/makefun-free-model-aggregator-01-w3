#!/usr/bin/env python3
"""Aggregator entry for free-model-aggregator-01, build "quick-failover".

One process fronts ten free-tier upstreams behind two client-facing dialects:
the OpenAI chat-completions dialect and the Anthropic messages dialect, in
both plain JSON and server-sent-event streaming shapes. For each incoming
request it picks the best-scoring free model that can actually handle what the
request asks for, and moves on to the next candidate whenever an upstream
turns the request away with a 402 or a 429. It also advertises its model
catalog to the harness, works through the harness workload, and leaves with
exit code zero once the harness reports that the work is finished.

Only the standard library is used. Nothing here depends on the clock: every
routing decision follows from the brief, from the request itself, and from
the answers upstreams have already given.
"""

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------------------------------------------------------------------
# Build dials: this wallet's copy tunes exactly these values.
# ---------------------------------------------------------------------------
VARIANT_NAME = "quick-failover"
PREFER_SAME_WIRE_BONUS = 2      # milli-rating lift when upstream speaks the client dialect
REFUSAL_LIMIT = 2               # turn-aways after which an upstream is treated as spent
LONG_CONTEXT_CHARS = 2500       # user text at or beyond this length counts as long-context
LONG_CONTEXT_MIN = 200000       # context window a long-context request insists on
ANTHROPIC_JSON_SAFE = False     # whether Anthropic-dialect calls demand a JSON-able model
STREAM_PIECE_CHARS = 16         # re-chunk streamed text to this width; 0 keeps upstream chunks
TIE_BREAK = "brief"             # tied scores resolve by "brief" (brief order) or "alpha" (name order)

FIXED_CREATED = 1767225600      # fixed creation stamp the fake upstreams also stamp
ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")
PORT = 8791

# ---------------------------------------------------------------------------
# Tiny HTTP caller built on urllib, with proxies switched off: trials hand us
# base URLs directly, so no proxy should ever get in the way.
# ---------------------------------------------------------------------------
_FETCHER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def do_request(verb, target, payload=None, extra_headers=None, wait=15):
    """A single HTTP round trip.

    Hands back (status code, headers mapping, raw body bytes). A failure at
    the connection level, where no answer arrives at all, comes back as
    status 0 with empty headers and body.
    """
    outgoing = None
    heads = dict(extra_headers or {})
    if payload is not None:
        outgoing = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        heads.setdefault("content-type", "application/json")
    call = urllib.request.Request(target, data=outgoing, headers=heads, method=verb)
    try:
        with _FETCHER.open(call, timeout=wait) as answer:
            return answer.status, dict(answer.headers.items()), answer.read()
    except urllib.error.HTTPError as failure:
        heads_out = dict(failure.headers.items()) if failure.headers else {}
        return failure.code, heads_out, failure.read()
    except Exception:
        return 0, {}, b""


def do_request_json(verb, target, payload=None, headers=None, wait=15):
    code, heads, blob = do_request(verb, target, payload, headers, wait)
    try:
        parsed = json.loads(blob.decode("utf-8")) if blob else None
    except Exception:
        parsed = None
    return code, heads, parsed


# ---------------------------------------------------------------------------
# The brief: the upstream directory, the quality table, and the workload
# manifest all ride inside the prompt file as JSON documents tucked into
# its prose.
# ---------------------------------------------------------------------------
def _find_doc(prose, schema_name):
    tag = '"schema":"' + schema_name + '"'
    spot = prose.find(tag)
    if spot < 0:
        tag = '"schema": "' + schema_name + '"'
        spot = prose.find(tag)
    if spot < 0:
        return None
    brace = prose.rfind("{", 0, spot)
    if brace < 0:
        return None
    try:
        found, _ = json.JSONDecoder().raw_decode(prose[brace:])
        return found
    except Exception:
        return None


class Upstream:
    __slots__ = ("id", "name", "wire", "auth", "key_env", "key", "path")

    def __init__(self, row):
        self.id = row["id"]
        self.name = row.get("name", row["id"])
        self.wire = row["wire"]
        self.auth = row.get("auth", "none")
        self.key_env = row.get("key_env", "")
        self.key = row.get("key", "")
        self.path = row["path"]

    def dial(self):
        return ENV_URL + self.path

    def auth_headers(self):
        if self.auth == "bearer":
            return {"authorization": "Bearer " + self.key}
        if self.auth == "x-api-key":
            return {"x-api-key": self.key}
        if self.auth == "x-goog-api-key":
            return {"x-goog-api-key": self.key}
        return {}


def read_brief():
    """Pull (upstreams, rating rows, manifest) out of the prompt file.

    An absent prompt file, or one that cannot be read, yields three empty
    collections rather than an error.
    """
    if not PROMPT_FILE:
        return [], [], []
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as handle:
            prose = handle.read()
    except Exception:
        return [], [], []
    directory = _find_doc(prose, "free-model-aggregator-01/providers/v1") or {}
    ratings = _find_doc(prose, "free-model-aggregator-01/ratings/v1") or {}
    workload = _find_doc(prose, "free-model-aggregator-01/workload/v1") or {}
    ups = [Upstream(row) for row in directory.get("providers", [])]
    return ups, ratings.get("ratings", []), workload.get("manifest", [])


# ---------------------------------------------------------------------------
# The catalog: ask each upstream for its own model list first, because that
# is ground truth, and only fall back to the brief's quality table when a
# probe comes back empty.
# ---------------------------------------------------------------------------
def _fetch_models(up):
    """[(model id, free?, context window)] off the upstream's catalog route.

    None when the probe fails outright, so the caller can fall back.
    """
    if up.wire == "gemini":
        target = up.dial() + "/v1beta/models"
    else:
        target = up.dial() + "/v1/models"
    # Slow catalog endpoints: give the probe a generous 15 seconds.
    code, _, doc = do_request_json("GET", target, headers=up.auth_headers(), wait=15)
    if code < 200 or code >= 300 or not isinstance(doc, dict):
        return None
    found = []
    if up.wire == "gemini":
        for entry in doc.get("models", []):
            mid = str(entry.get("name", "")).removeprefix("models/")
            if not mid:
                continue
            window = entry.get("context_length") or entry.get("inputTokenLimit")
            found.append((mid, entry.get("free") is True, window))
    else:
        for entry in doc.get("data", []):
            if isinstance(entry.get("id"), str) and entry["id"]:
                found.append((entry["id"], entry.get("free") is True, entry.get("context_length")))
    return found or None


def make_catalog(ups, rating_rows):
    by_upstream = {}
    for row in rating_rows:
        by_upstream.setdefault(row["provider"], []).append(
            (row["model"], row.get("free") is True, row.get("context_length")))
    catalog = {"providers": []}
    for up in ups:
        probed = _fetch_models(up) if ENV_URL else None
        models = probed if probed is not None else by_upstream.get(up.id, [])
        catalog["providers"].append({
            "id": up.id,
            "base_url": up.path,
            "key_env": up.key_env,
            "wire": up.wire,
            "models": [
                {"id": mid, "free": is_free, "context_length": window}
                for mid, is_free, window in models
            ],
        })
    return catalog


def dump_catalog(catalog):
    return json.dumps(catalog)

# ---------------------------------------------------------------------------
# The canonical request: one dialect-neutral view of what the client asked
# for, with nothing thrown away.
# ---------------------------------------------------------------------------
def _picture_part(link):
    if isinstance(link, str) and link.startswith("data:") and ";base64," in link:
        media = link[5:link.index(";base64,")]
        return {"type": "image", "media_type": media, "data": link.split(";base64,", 1)[1]}
    return {"type": "image_ref", "url": link}


def _join_text_blocks(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        bits = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                bits.append(block["text"])
        return "\n".join(bits)
    return ""


class NormalReq:
    """Dialect-neutral request.

    system holds a string or None; messages is a list of {role, parts} where
    each part is {type:text, text} or {type:image, media_type, data} or
    {type:image_ref, url}; tools is a list of {name, description,
    parameters}; the untouched client body is kept in raw.
    """

    def __init__(self, dialect, body):
        self.api = dialect
        self.raw = body
        self.stream = body.get("stream") is True
        self.max_tokens = None
        self.stops = []
        self.json_mode = False
        self.response_format = None
        self.system = None
        self.messages = []
        self.tools = []
        self.tool_choice = None
        if dialect == "openai":
            self._load_openai(body)
        else:
            self._load_anthropic(body)

    def _load_openai(self, body):
        cap = body.get("max_tokens")
        if not isinstance(cap, (int, float)):
            cap = body.get("max_completion_tokens")
        self.max_tokens = cap if isinstance(cap, (int, float)) else None
        halt = body.get("stop")
        if isinstance(halt, str):
            self.stops = [halt]
        elif isinstance(halt, list):
            self.stops = [item for item in halt if isinstance(item, str)]
        shape = body.get("response_format")
        if isinstance(shape, dict):
            self.response_format = shape
            self.json_mode = shape.get("type") in ("json_object", "json_schema")
        self.tool_choice = body.get("tool_choice")
        for tool in body.get("tools") or []:
            fn = tool.get("function") if isinstance(tool, dict) else None
            if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                self.tools.append({
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters") or {"type": "object"},
                })
        sys_bits = []
        for msg in body.get("messages") or []:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                text = _join_text_blocks(content)
                if text:
                    sys_bits.append(text)
                continue
            parts = []
            if isinstance(content, str):
                if content:
                    parts.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        parts.append({"type": "text", "text": block["text"]})
                    elif block.get("type") == "image_url" and isinstance(block.get("image_url"), dict):
                        parts.append(_picture_part(block["image_url"].get("url", "")))
            self.messages.append({"role": role or "user", "parts": parts})
        self.system = "\n".join(sys_bits) if sys_bits else None

    def _load_anthropic(self, body):
        cap = body.get("max_tokens")
        self.max_tokens = cap if isinstance(cap, (int, float)) else None
        halts = body.get("stop_sequences")
        if isinstance(halts, list):
            self.stops = [item for item in halts if isinstance(item, str)]
        self.tool_choice = body.get("tool_choice")
        sys_in = body.get("system")
        if isinstance(sys_in, str):
            self.system = sys_in or None
        elif isinstance(sys_in, list):
            text = _join_text_blocks(sys_in)
            self.system = text or None
        for tool in body.get("tools") or []:
            if isinstance(tool, dict) and isinstance(tool.get("name"), str):
                self.tools.append({
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema") or {"type": "object"},
                })
        for msg in body.get("messages") or []:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            parts = []
            if isinstance(content, str):
                if content:
                    parts.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        parts.append({"type": "text", "text": block["text"]})
                    elif block.get("type") == "image" and isinstance(block.get("source"), dict):
                        src = block["source"]
                        if src.get("type") == "base64" and isinstance(src.get("data"), str):
                            parts.append({"type": "image", "media_type": src.get("media_type", "image/png"), "data": src["data"]})
                        elif src.get("type") == "url":
                            parts.append({"type": "image_ref", "url": src.get("url", "")})
            self.messages.append({"role": msg.get("role") or "user", "parts": parts})

    # -- what this request demands, read straight off the request itself ----
    @property
    def wants_tools(self):
        return len(self.tools) > 0

    @property
    def wants_vision(self):
        return any(p["type"] in ("image", "image_ref") for m in self.messages for p in m["parts"])

    @property
    def last_user_text(self):
        for msg in reversed(self.messages):
            if msg["role"] == "user":
                return "\n".join(p["text"] for p in msg["parts"] if p["type"] == "text")
        return ""


# ---------------------------------------------------------------------------
# Turning the canonical request back into each upstream dialect.
# ---------------------------------------------------------------------------
def _as_gemini_payload(req):
    turns = []
    for msg in req.messages:
        bits = []
        for p in msg["parts"]:
            if p["type"] == "text":
                bits.append({"text": p["text"]})
            elif p["type"] == "image":
                bits.append({"inlineData": {"mimeType": p["media_type"], "data": p["data"]}})
            else:
                bits.append({"fileData": {"fileUri": p["url"]}})
        turns.append({"role": "model" if msg["role"] == "assistant" else "user", "parts": bits})
    payload = {"contents": turns}
    if req.system is not None:
        payload["systemInstruction"] = {"parts": [{"text": req.system}]}
    if req.tools:
        payload["tools"] = [{"functionDeclarations": [
            {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}
            for t in req.tools]}]
    cfg = {}
    if req.max_tokens is not None:
        cfg["maxOutputTokens"] = req.max_tokens
    if req.stops:
        cfg["stopSequences"] = list(req.stops)
    if req.json_mode:
        cfg["responseMimeType"] = "application/json"
    if cfg:
        payload["generationConfig"] = cfg
    return payload


def _as_openai_msgs(req):
    msgs = []
    if req.system is not None:
        msgs.append({"role": "system", "content": req.system})
    for msg in req.messages:
        parts = msg["parts"]
        if all(p["type"] == "text" for p in parts):
            msgs.append({"role": msg["role"], "content": "\n".join(p["text"] for p in parts)})
            continue
        content = []
        for p in parts:
            if p["type"] == "text":
                content.append({"type": "text", "text": p["text"]})
            elif p["type"] == "image":
                content.append({"type": "image_url", "image_url": {
                    "url": "data:%s;base64,%s" % (p["media_type"], p["data"])}})
            else:
                content.append({"type": "image_url", "image_url": {"url": p["url"]}})
        msgs.append({"role": msg["role"], "content": content})
    return msgs


def _as_anthropic_msgs(req):
    msgs = []
    for msg in req.messages:
        parts = msg["parts"]
        if all(p["type"] == "text" for p in parts):
            msgs.append({"role": msg["role"], "content": "\n".join(p["text"] for p in parts)})
            continue
        content = []
        for p in parts:
            if p["type"] == "text":
                content.append({"type": "text", "text": p["text"]})
            elif p["type"] == "image":
                content.append({"type": "image", "source": {
                    "type": "base64", "media_type": p["media_type"], "data": p["data"]}})
            else:
                content.append({"type": "image", "source": {"type": "url", "url": p["url"]}})
        msgs.append({"role": msg["role"], "content": content})
    return msgs


def _translate_tool_choice(wire, req):
    choice = req.tool_choice
    if choice is None:
        return None, False
    if wire == "openai":
        return choice, True
    if wire == "anthropic":
        if isinstance(choice, str):
            return {"type": choice if choice in ("auto", "any", "none") else "auto"}, True
        if isinstance(choice, dict):
            if choice.get("type") == "function" and isinstance(choice.get("function"), dict):
                return {"type": "tool", "name": choice["function"].get("name")}, True
            if choice.get("type") in ("auto", "any", "none"):
                return {"type": choice["type"]}, True
        return None, False
    return None, False

def make_upstream_request(up, model_id, req):
    """(url, headers, body dict) for a single inference call on this upstream."""
    heads = up.auth_headers()
    if up.wire == "openai":
        if req.api == "openai":
            payload = json.loads(json.dumps(req.raw))  # the client's own body, verbatim
            payload["model"] = model_id
        else:
            payload = {"model": model_id, "messages": _as_openai_msgs(req)}
            if req.max_tokens is not None:
                payload["max_tokens"] = req.max_tokens
            if req.stops:
                payload["stop"] = list(req.stops)
            if req.tools:
                payload["tools"] = [{"type": "function", "function": {
                    "name": t["name"], "description": t["description"],
                    "parameters": t["parameters"]}} for t in req.tools]
            picked, usable = _translate_tool_choice("openai", req)
            if usable:
                payload["tool_choice"] = picked
            if req.stream:
                payload["stream"] = True
        if req.stream:
            payload["stream"] = True
            stream_opts = dict(payload.get("stream_options") or {})
            stream_opts["include_usage"] = True
            payload["stream_options"] = stream_opts
        return up.dial() + "/v1/chat/completions", heads, payload
    if up.wire == "anthropic":
        heads["anthropic-version"] = "2023-06-01"
        if req.api == "anthropic":
            payload = json.loads(json.dumps(req.raw))
            payload["model"] = model_id
        else:
            payload = {"model": model_id, "messages": _as_anthropic_msgs(req)}
            if req.system is not None:
                payload["system"] = req.system
            if req.max_tokens is not None:
                payload["max_tokens"] = req.max_tokens
            if req.stops:
                payload["stop_sequences"] = list(req.stops)
            if req.tools:
                payload["tools"] = [{"name": t["name"], "description": t["description"],
                                     "input_schema": t["parameters"]} for t in req.tools]
            picked, usable = _translate_tool_choice("anthropic", req)
            if usable:
                payload["tool_choice"] = picked
            if req.stream:
                payload["stream"] = True
        return up.dial() + "/v1/messages", heads, payload
    # gemini
    payload = _as_gemini_payload(req)
    verb = "streamGenerateContent" if req.stream else "generateContent"
    target = up.dial() + "/v1beta/models/%s:%s" % (
        urllib.parse.quote(model_id, safe=""), verb)
    if req.stream:
        target += "?alt=sse"
    return target, heads, payload


# ---------------------------------------------------------------------------
# Reading an upstream answer back into (text, stream pieces, finish reason,
# token usage as (in, out)).
# ---------------------------------------------------------------------------
def _split_sse(blob):
    events = []
    normalized = blob.decode("utf-8", "replace").replace("\r\n", "\n")
    for block in normalized.split("\n\n"):
        kind = None
        payload_lines = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                kind = line[6:].strip()
            elif line.startswith("data:"):
                payload_lines.append(line[5:].lstrip(" "))
        if payload_lines:
            events.append((kind, "\n".join(payload_lines)))
    return events


FINISH_FOR_OPENAI = {"stop": "stop", "end_turn": "stop", "STOP": "stop",
                     "max_tokens": "length", "MAX_TOKENS": "length",
                     "tool_use": "tool_calls", "refusal": "content_filter", "SAFETY": "content_filter"}
FINISH_FOR_ANTHROPIC = {"stop": "end_turn", "STOP": "end_turn", "end_turn": "end_turn",
                        "length": "max_tokens", "MAX_TOKENS": "max_tokens",
                        "tool_calls": "tool_use", "function_call": "tool_use",
                        "content_filter": "refusal", "SAFETY": "refusal"}


def read_reply(wire, blob, streamed):
    """Decode one upstream answer into {text, pieces, stop, usage}, or None.

    None means the bytes could not be understood at all, so the router
    should move on to the next candidate rather than fail the request.
    """
    try:
        if wire == "openai":
            if not streamed:
                doc = json.loads(blob.decode("utf-8"))
                picked = doc["choices"][0]
                body_text = picked["message"]["content"]
                if isinstance(body_text, list):
                    body_text = "".join(bit.get("text", "") for bit in body_text if isinstance(bit, dict))
                counts = doc.get("usage") or {}
                return {"text": body_text, "pieces": [body_text], "stop": picked.get("finish_reason"),
                        "usage": (counts.get("prompt_tokens"), counts.get("completion_tokens"))}
            text, pieces, stop, counts = "", [], None, None
            for _, data in _split_sse(blob):
                if data.strip() == "[DONE]":
                    continue
                part = json.loads(data)
                if isinstance(part.get("usage"), dict):
                    counts = (part["usage"].get("prompt_tokens"), part["usage"].get("completion_tokens"))
                if part.get("choices"):
                    picked = part["choices"][0]
                    delta = picked.get("delta") or {}
                    if isinstance(delta.get("content"), str) and delta["content"]:
                        pieces.append(delta["content"])
                        text += delta["content"]
                    if isinstance(picked.get("finish_reason"), str):
                        stop = picked["finish_reason"]
            return {"text": text, "pieces": pieces, "stop": stop, "usage": counts}
        if wire == "anthropic":
            if not streamed:
                doc = json.loads(blob.decode("utf-8"))
                text = "".join(bit.get("text", "") for bit in doc.get("content", [])
                               if isinstance(bit, dict) and bit.get("type") == "text")
                counts = doc.get("usage") or {}
                return {"text": text, "pieces": [text], "stop": doc.get("stop_reason"),
                        "usage": (counts.get("input_tokens"), counts.get("output_tokens"))}
            text, pieces, stop = "", [], None
            tokens_in, tokens_out = None, None
            for kind, data in _split_sse(blob):
                doc = json.loads(data)
                if kind == "message_start":
                    tokens_in = (doc.get("message") or {}).get("usage", {}).get("input_tokens")
                elif kind == "content_block_delta":
                    delta = doc.get("delta") or {}
                    if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                        pieces.append(delta["text"])
                        text += delta["text"]
                elif kind == "message_delta":
                    stop = (doc.get("delta") or {}).get("stop_reason") or stop
                    tokens_out = (doc.get("usage") or {}).get("output_tokens", tokens_out)
            usage = (tokens_in, tokens_out) if tokens_in is not None or tokens_out is not None else None
            return {"text": text, "pieces": pieces, "stop": stop, "usage": usage}
        # gemini
        if not streamed:
            doc = json.loads(blob.decode("utf-8"))
            cand = doc["candidates"][0]
            text = "".join(bit.get("text", "") for bit in cand["content"]["parts"] if isinstance(bit, dict))
            meta = doc.get("usageMetadata") or {}
            return {"text": text, "pieces": [text], "stop": cand.get("finishReason"),
                    "usage": (meta.get("promptTokenCount"), meta.get("candidatesTokenCount"))}
        text, pieces, stop, counts = "", [], None, None
        for _, data in _split_sse(blob):
            doc = json.loads(data)
            for cand in doc.get("candidates", []):
                for bit in (cand.get("content") or {}).get("parts", []):
                    if isinstance(bit, dict) and isinstance(bit.get("text"), str) and bit["text"]:
                        pieces.append(bit["text"])
                        text += bit["text"]
                if isinstance(cand.get("finishReason"), str):
                    stop = cand["finishReason"]
            meta = doc.get("usageMetadata")
            if isinstance(meta, dict):
                counts = (meta.get("promptTokenCount"), meta.get("candidatesTokenCount"))
        return {"text": text, "pieces": pieces, "stop": stop, "usage": counts}
    except Exception:
        return None


def _chop_pieces(reply):
    chunks = [piece for piece in reply["pieces"] if piece]
    if STREAM_PIECE_CHARS > 0:
        full = reply["text"]
        chunks = [full[i:i + STREAM_PIECE_CHARS] for i in range(0, len(full), STREAM_PIECE_CHARS)] or [""]
    return chunks or [""]


# ---------------------------------------------------------------------------
# Answering the client back in the dialect it spoke.
# ---------------------------------------------------------------------------
def build_client_reply(req, model_id, reply):
    """(content type, body bytes) for the client's own dialect, streamed if asked."""
    counts = reply["usage"] or (None, None)
    if req.api == "openai":
        finish = FINISH_FOR_OPENAI.get(reply["stop"], "stop")
        if not req.stream:
            doc = {"id": "chatcmpl-agg", "object": "chat.completion", "created": FIXED_CREATED,
                   "model": model_id,
                   "choices": [{"index": 0, "message": {"role": "assistant", "content": reply["text"]},
                                "finish_reason": finish, "logprobs": None}],
                   "usage": {"prompt_tokens": counts[0], "completion_tokens": counts[1],
                             "total_tokens": (counts[0] or 0) + (counts[1] or 0)}}
            return "application/json", json.dumps(doc).encode("utf-8")
        frames = []

        def piece_chunk(delta, done):
            return {"id": "chatcmpl-agg", "object": "chat.completion.chunk", "created": FIXED_CREATED,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": done}]}

        frames.append("data: " + json.dumps(piece_chunk({"role": "assistant", "content": ""}, None)) + "\n\n")
        for bit in _chop_pieces(reply):
            frames.append("data: " + json.dumps(piece_chunk({"content": bit}, None)) + "\n\n")
        frames.append("data: " + json.dumps(piece_chunk({}, finish)) + "\n\n")
        if counts[0] is not None:
            frames.append("data: " + json.dumps({
                "id": "chatcmpl-agg", "object": "chat.completion.chunk", "created": FIXED_CREATED,
                "model": model_id, "choices": [],
                "usage": {"prompt_tokens": counts[0], "completion_tokens": counts[1],
                          "total_tokens": (counts[0] or 0) + (counts[1] or 0)}}) + "\n\n")
        frames.append("data: [DONE]\n\n")
        return "text/event-stream", "".join(frames).encode("utf-8")
    # anthropic
    finish = FINISH_FOR_ANTHROPIC.get(reply["stop"], "end_turn")
    if not req.stream:
        doc = {"id": "msg_agg", "type": "message", "role": "assistant", "model": model_id,
               "content": [{"type": "text", "text": reply["text"]}],
               "stop_reason": finish, "stop_sequence": None,
               "usage": {"input_tokens": counts[0], "output_tokens": counts[1]}}
        return "application/json", json.dumps(doc).encode("utf-8")
    frames = []

    def sse_frame(kind, doc):
        return "event: %s\ndata: %s\n\n" % (kind, json.dumps(doc))

    frames.append(sse_frame("message_start", {
        "type": "message_start",
        "message": {"id": "msg_agg", "type": "message", "role": "assistant", "model": model_id,
                    "content": [], "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": counts[0], "output_tokens": 0}}}))
    frames.append(sse_frame("content_block_start", {
        "type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}))
    for bit in _chop_pieces(reply):
        frames.append(sse_frame("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": bit}}))
    frames.append(sse_frame("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(sse_frame("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": finish, "stop_sequence": None},
        "usage": {"output_tokens": counts[1]}}))
    frames.append(sse_frame("message_stop", {"type": "message_stop"}))
    return "text/event-stream", "".join(frames).encode("utf-8")

# ---------------------------------------------------------------------------
# Routing: order the free models that fit the request by the brief's rating,
# try them one after another, and keep a tally of turn-aways as they happen.
# ---------------------------------------------------------------------------
class Chooser:
    def __init__(self, ups, rating_rows):
        self.ups = {u.id: u for u in ups}
        self.order = [u.id for u in ups]
        self.table = {}
        for row in rating_rows:
            self.table[(row["provider"], row["model"])] = row
        self.calls = {pid: 0 for pid in self.order}
        self.turned_away = {pid: 0 for pid in self.order}
        self.spent = set()

    def rank_models(self, req):
        wants_json = req.json_mode or (req.api == "anthropic" and ANTHROPIC_JSON_SAFE)
        is_long = len(req.last_user_text) >= LONG_CONTEXT_CHARS
        scored = []
        for slot, pid in enumerate(self.order):
            up = self.ups[pid]
            for (prov_id, model_id), row in self.table.items():
                if prov_id != pid or row.get("free") is not True:
                    continue
                if req.wants_tools and not row.get("tools"):
                    continue
                if req.wants_vision and not row.get("vision"):
                    continue
                if wants_json and not row.get("json"):
                    continue
                if is_long and (row.get("context_length") or 0) < LONG_CONTEXT_MIN:
                    continue
                points = row.get("rating_milli", 0)
                if up.wire == req.api:
                    points += PREFER_SAME_WIRE_BONUS
                scored.append((-points, slot, up, model_id))
        if TIE_BREAK == "alpha":
            scored.sort(key=lambda entry: (entry[0], entry[2].id, entry[3]))
        else:
            scored.sort(key=lambda entry: (entry[0], entry[1]))
        return [(up, model_id) for _, _, up, model_id in scored]

    def dispatch(self, req):
        for up, model_id in self.rank_models(req):
            if up.id in self.spent:
                continue
            target, heads, payload = make_upstream_request(up, model_id, req)
            code, _, blob = do_request("POST", target, payload, heads, wait=20)
            self.calls[up.id] += 1
            if 200 <= code < 300:
                reply = read_reply(up.wire, blob, req.stream)
                if reply is not None:
                    return model_id, reply
                continue
            if code in (402, 429):
                # A turn-away burns one strike against this upstream only; the
                # request itself is still live and the next model gets a try.
                self.turned_away[up.id] += 1
                if self.turned_away[up.id] >= REFUSAL_LIMIT:
                    self.spent.add(up.id)
                continue
            # anything else (a 400, a 5xx, silence): try the next model
        return None


# ---------------------------------------------------------------------------
# The HTTP front door the harness client talks to.
# ---------------------------------------------------------------------------
RUNTIME = {"router": None, "catalog": None}


def render_model_list(headers):
    catalog = RUNTIME["catalog"] or {"providers": []}
    seen = []
    for up in catalog["providers"]:
        for model in up["models"]:
            if model["free"] and model["id"] not in seen:
                seen.append(model["id"])
    if headers.get("x-api-key") is not None:
        doc = {"data": [{"type": "model", "id": mid, "display_name": mid,
                         "created_at": "2026-01-01T00:00:00Z"} for mid in seen],
               "has_more": False,
               "first_id": seen[0] if seen else None,
               "last_id": seen[-1] if seen else None}
    else:
        doc = {"object": "list",
               "data": [{"id": mid, "object": "model", "created": FIXED_CREATED,
                         "owned_by": "aggregator"} for mid in seen]}
    return json.dumps(doc).encode("utf-8")


def make_error(dialect, message):
    if dialect == "anthropic":
        return json.dumps({"type": "error", "error": {"type": "api_error", "message": message}}).encode("utf-8")
    return json.dumps({"error": {"message": message, "type": "api_error",
                                 "param": None, "code": "no_provider"}}).encode("utf-8")


class Gateway(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _write_out(self, code, mime, payload):
        try:
            self.send_response(code)
            self.send_header("content-type", mime)
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path.split("?")[0] == "/v1/models":
            self._write_out(200, "application/json", render_model_list(self.headers))
        else:
            self._write_out(404, "application/json", make_error("openai", "not found"))

    def do_POST(self):
        route = self.path.split("?")[0]
        if route == "/v1/chat/completions":
            dialect = "openai"
        elif route == "/v1/messages":
            dialect = "anthropic"
        else:
            self._write_out(404, "application/json", make_error("openai", "not found"))
            return
        try:
            size = int(self.headers.get("content-length") or 0)
        except ValueError:
            size = 0
        raw = self.rfile.read(size) if size > 0 else b""
        try:
            body = json.loads(raw.decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("body is not an object")
        except Exception:
            self._write_out(400, "application/json", make_error(dialect, "the request body is not one JSON object"))
            return
        req = NormalReq(dialect, body)
        served = RUNTIME["router"].dispatch(req)
        if served is None:
            self._write_out(502, "application/json", make_error(dialect, "no free model could serve this request"))
            return
        model_id, reply = served
        mime, payload = build_client_reply(req, model_id, reply)
        self._write_out(200, mime, payload)


# ---------------------------------------------------------------------------
# Trial lifecycle: probe the catalog, announce ourselves, serve until the
# harness says the work is done, then leave quietly with exit code zero.
# ---------------------------------------------------------------------------
def main(argv):
    ups, rating_rows, _ = read_brief()
    if "--list" in argv:
        if not ups:
            print('{"providers":[]}')
            return 0
        print(dump_catalog(make_catalog(ups, rating_rows)))
        return 0
    catalog = make_catalog(ups, rating_rows)
    RUNTIME["catalog"] = catalog
    RUNTIME["router"] = Chooser(ups, rating_rows)
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Gateway)
    server.daemon_threads = True
    import threading
    import time
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
    worker.daemon = True
    worker.start()
    if ENV_URL:
        for _ in range(30):
            code, _, doc = do_request_json("POST", ENV_URL + "/register",
                                           {"protocol_version": 1, "port": PORT, "catalog": catalog},
                                           wait=5)
            if code == 200 and isinstance(doc, dict) and doc.get("ok") is True:
                break
            time.sleep(0.5)
        misses = 0
        waited = 0.0
        while waited < 170.0:
            code, _, doc = do_request_json("GET", ENV_URL + "/work", wait=5)
            if code == 200 and isinstance(doc, dict):
                misses = 0
                if doc.get("done") is True:
                    break
            else:
                misses += 1
                if misses > 40:
                    break
            time.sleep(0.2)
            waited += 0.2
    else:
        worker.join()
    server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
