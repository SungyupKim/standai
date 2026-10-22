"""Sends a KakaoTalk "나에게 보내기" (send-to-me) notification -- used to ping when a
training round finishes. Token file lives in checkpoints/ (already gitignored), never
committed. REST API key is passed in, never hardcoded here.

One-time setup:
    python kakao_notify.py auth --rest-api-key KEY --redirect-uri URI --code CODE
      -- exchanges the OAuth code for access/refresh tokens, saves them locally.

Then just:
    python kakao_notify.py send "message text"
      -- sends a message, auto-refreshing the access token first if it's expired.

Or from another script:
    from kakao_notify import send_message
    send_message("round 5 done")
"""
import argparse
import json
import re
import pathlib
import time
import urllib.parse
import urllib.request

TOKEN_PATH = pathlib.Path(__file__).resolve().parent.parent / "checkpoints" / "kakao_token.json"


def _post(url, data):
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def exchange_code(rest_api_key, redirect_uri, code, client_secret=None):
    data = {
        "grant_type": "authorization_code",
        "client_id": rest_api_key,
        "redirect_uri": redirect_uri,
        "code": code,
    }
    if client_secret:
        data["client_secret"] = client_secret
    result = _post("https://kauth.kakao.com/oauth/token", data)
    result["rest_api_key"] = rest_api_key
    if client_secret:
        result["client_secret"] = client_secret
    result["obtained_at"] = time.time()
    TOKEN_PATH.write_text(json.dumps(result))
    print(f"saved tokens to {TOKEN_PATH}")
    return result


def _load_tokens():
    if not TOKEN_PATH.exists():
        raise RuntimeError(f"no token file at {TOKEN_PATH} -- run the 'auth' subcommand first")
    return json.loads(TOKEN_PATH.read_text())


def _refresh(tokens):
    data = {
        "grant_type": "refresh_token",
        "client_id": tokens["rest_api_key"],
        "refresh_token": tokens["refresh_token"],
    }
    if tokens.get("client_secret"):
        data["client_secret"] = tokens["client_secret"]
    result = _post("https://kauth.kakao.com/oauth/token", data)
    tokens["access_token"] = result["access_token"]
    tokens["expires_in"] = result.get("expires_in", tokens.get("expires_in"))
    if "refresh_token" in result:  # Kakao sometimes rotates the refresh token too
        tokens["refresh_token"] = result["refresh_token"]
    tokens["obtained_at"] = time.time()
    TOKEN_PATH.write_text(json.dumps(tokens))
    return tokens


def _send_one(text):
    tokens = _load_tokens()
    if time.time() - tokens["obtained_at"] > tokens.get("expires_in", 21599) - 60:
        tokens = _refresh(tokens)

    template = json.dumps({
        "object_type": "text",
        "text": text,
        "link": {},
    })
    body = urllib.parse.urlencode({"template_object": template}).encode()
    req = urllib.request.Request(
        "https://kapi.kakao.com/v2/api/talk/memo/default/send", data=body, method="POST",
    )
    req.add_header("Authorization", f"Bearer {tokens['access_token']}")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        # one retry after a forced refresh, in case the token was invalid rather than just expired
        tokens = _refresh(tokens)
        req.headers["Authorization"] = f"Bearer {tokens['access_token']}"
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())



# ---- readability: tidy a message and split it into <= LIMIT-character parts (Kakao's text template shows about 200 characters per message) ----
LIMIT = 190


def _shorten_names(line):
    line = re.sub(r"ppo_floor_v12_[0-9_]*", "v12", line)
    line = re.sub(r"floor_supine_(\w+?)\.zi?p?", r"\1", line)           # file names cut by `cut -c` in the shell scripts
    line = re.sub(r"(\w+)\.zip", r"\1", line)
    line = line.replace("handover_supine_s2", "s2").replace("best_handover_prone", "인계정책").replace("v_gate None flip 0: ", "")
    return line


def prettify(text):
    """Controller-evaluation lines -> '20초 서기 40% (80/200)'; '[tag] text' -> '📌 tag' headers; long file names shortened. Other lines untouched."""
    out = []
    for raw in text.splitlines():
        line = _shorten_names(raw.rstrip())
        m = re.search(r"standing at 20 s (\d+)/(\d+)\s*\((?:10 s (\d+), )?handed over (\d+)\)", line)
        if m:
            k, n, h = int(m.group(1)), int(m.group(2)), int(m.group(4))
            head = line[:m.start()].replace("floor ", "").replace("hand-over ", "").replace("supine ", "").replace("prone ", "")
            head = re.sub(r"\s+", " ", head).strip(" :")
            line = f" · {head}: 20초 서기 {100 * k / n:.0f}% ({k}/{n}), 인계 {100 * h / n:.0f}%"
        else:
            m = re.match(r"\[(.+?)\]\s*(.*)", line)
            if m:
                line = "📌 " + m.group(1) + (("\n" + m.group(2)) if m.group(2) else "")
        out.append(line)
    return "\n".join(out)


def split_message(text, limit=LIMIT):
    """Split on line breaks into parts of at most `limit` characters; parts are numbered '(i/n)' when there is more than one."""
    parts, cur = [], ""
    for line in text.splitlines():
        while len(line) > limit:                       # a single very long line: cut at a space
            cut = line.rfind(" ", 0, limit) if " " in line[:limit] else limit
            cut = cut if cut > 20 else limit
            if cur: parts.append(cur); cur = ""
            parts.append(line[:cut]); line = line[cut:].lstrip()
        if len(cur) + len(line) + 1 > limit and cur:
            parts.append(cur); cur = line
        else:
            cur = (cur + "\n" + line) if cur else line
    if cur: parts.append(cur)
    if len(parts) > 1:
        parts = [f"({i + 1}/{len(parts)})\n{p}" for i, p in enumerate(parts)]
    return parts or [""]


def send_message(text):
    """Tidies the text (prettify) and sends it in parts of at most LIMIT characters. KAKAO_RAW=1 sends it unchanged."""
    import os
    if os.environ.get("KAKAO_RAW") == "1":
        return _send_one(text)
    last = None
    for part in split_message(prettify(text)):
        last = _send_one(part)
        time.sleep(0.4)
    return last


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    auth_p = sub.add_parser("auth")
    auth_p.add_argument("--rest-api-key", required=True)
    auth_p.add_argument("--redirect-uri", required=True)
    auth_p.add_argument("--code", required=True)
    auth_p.add_argument("--client-secret", default=None)

    send_p = sub.add_parser("send")
    send_p.add_argument("text")

    args = parser.parse_args()
    if args.cmd == "auth":
        exchange_code(args.rest_api_key, args.redirect_uri, args.code, args.client_secret)
    elif args.cmd == "send":
        result = send_message(args.text)
        print(result)


if __name__ == "__main__":
    main()
