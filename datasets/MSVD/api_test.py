import os
import requests

API_URL = "https://globalai.vip/v1/chat/completions"
MODEL = "gpt-4.1-nano"
KEYS_FILE = ".keys"


def load_keys(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Keys file not found: {path}")

    keys = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # 去掉注释和空行
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 支持在一行末尾写注释：key # comment
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            keys.append(line)
    return keys


def mask_key(k: str) -> str:
    if len(k) <= 10:
        return "*" * len(k)
    return f"{k[:6]}***{k[-4:]}"


def test_key(key: str, index: int):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你好，这是一条测试请求，请只回复：OK"},
        ],
        "temperature": 0.3,
        "max_tokens": 16,
    }

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    print(f"[{index}] Testing key: {mask_key(key)}")

    try:
        resp = requests.post(API_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        print(f"[{index}] -> ERROR in HTTP request: {e}")
        return

    try:
        data = resp.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        content_preview = content.strip().replace("\n", " ")[:80]
        print(f"[{index}] -> OK, reply: {content_preview}")
    except Exception as e:
        text_preview = resp.text[:200].replace("\n", " ")
        print(f"[{index}] -> ERROR parsing JSON: {e}; raw response: {text_preview}")


def main():
    try:
        keys = load_keys(KEYS_FILE)
    except FileNotFoundError as e:
        print(str(e))
        return

    if not keys:
        print(f"No keys found in {KEYS_FILE}")
        return

    print(f"Loaded {len(keys)} key(s) from {KEYS_FILE}")

    for i, key in enumerate(keys, start=1):
        test_key(key, i)


if __name__ == "__main__":
    main()
