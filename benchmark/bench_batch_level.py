#!/usr/bin/env python3
"""Send N concurrent requests to MoLink or vLLM and report timing.

Usage:
    python bench_batch_level.py --url http://localhost:8080/generate --type molink
    python bench_batch_level.py --url http://localhost:8080/v1/completions --type vllm \\
        --model /gxq/Qwen3-14B --concurrent 3
"""

import argparse
import asyncio
import json
import random
import string
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp

# ── Tokenizer (same as benchmark_client.py) ────────────────────────────────

_DEFAULT_TOKENIZER_PATH = "/home/emnets-2/gxq/Qwen3-14B"
_FALLBACK_CHARS_PER_TOKEN = 6.5
_tokenizer_instance = None

_PROMPT_BLOCKS = [
    "Artificial intelligence has fundamentally transformed the way we interact with "
    "technology in the modern era. From natural language processing to computer vision, "
    "machine learning models have achieved remarkable capabilities that were once thought "
    "to be exclusively human domains. The development of large language models represents "
    "a significant milestone in this journey, enabling machines to understand and generate "
    "human language with unprecedented fluency and coherence.",

    "The history of computing stretches back centuries, from Charles Babbage's analytical "
    "engine to the modern quantum computers being developed today. Each generation of "
    "computing technology has brought exponential increases in processing power and "
    "efficiency. The invention of the transistor in the mid-twentieth century revolutionized "
    "electronics and laid the groundwork for the integrated circuits that power our modern "
    "world.",

    "Climate change represents one of the most significant challenges facing humanity in "
    "the twenty-first century. Rising global temperatures are driving shifts in weather "
    "patterns, rising sea levels, and increasing frequency of extreme weather events. "
    "Scientists around the world are working to develop renewable energy sources, carbon "
    "capture technologies, and sustainable agricultural practices to mitigate these effects.",
]
_PROMPT_TEXT = " ".join(_PROMPT_BLOCKS)


def get_tokenizer(path: str | None = None):
    global _tokenizer_instance
    if _tokenizer_instance is not None:
        return _tokenizer_instance
    try:
        from transformers import AutoTokenizer
        p = path or _DEFAULT_TOKENIZER_PATH
        _tokenizer_instance = AutoTokenizer.from_pretrained(p)
    except Exception:
        _tokenizer_instance = None
    return _tokenizer_instance


def count_tokens(text: str) -> int:
    tok = get_tokenizer()
    if tok is not None:
        return len(tok.encode(text))
    return int(len(text) / _FALLBACK_CHARS_PER_TOKEN)


def truncate_to_tokens(text: str, n: int) -> str:
    tok = get_tokenizer()
    if tok is not None:
        ids = tok.encode(text)[:n]
        return tok.decode(ids)
    target_chars = int(n * _FALLBACK_CHARS_PER_TOKEN)
    return text[:target_chars]


def generate_prompt(target_tokens: int, request_index: int) -> str:
    rng = random.Random(request_index)
    padding = ''.join(rng.choices(string.ascii_lowercase, k=120))
    prefix = f"<|req_{request_index}|>{padding}\n\n"
    repeats = (target_tokens * int(_FALLBACK_CHARS_PER_TOKEN) * 2) // len(_PROMPT_TEXT) + 1
    long_text = prefix + (_PROMPT_TEXT + " ") * repeats
    return truncate_to_tokens(long_text, target_tokens)


# ── Request functions ──────────────────────────────────────────────────────

async def _read_lines(response):
    buf = b""
    async for chunk in response.content.iter_any():
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            decoded = line.decode("utf-8").strip()
            if decoded:
                yield decoded


async def bench_one_molink(session: aiohttp.ClientSession, url: str,
                           prompt: str, max_tokens: int, req_idx: int) -> dict:
    payload = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "stream": True}
    start = time.monotonic()
    ttft = None
    prev_text_len = len(prompt)
    final_text = ""
    try:
        timeout = aiohttp.ClientTimeout(total=600)
        async with session.post(url, json=payload, timeout=timeout) as resp:
            if resp.status != 200:
                body = await resp.text()
                return {"success": False, "error": f"HTTP {resp.status}: {body[:200]}", "index": req_idx}
            async for line in _read_lines(resp):
                now = time.monotonic()
                try:
                    data = json.loads(line)
                    text = data.get("text", [""])[0]
                except (json.JSONDecodeError, IndexError):
                    continue
                if len(text) > prev_text_len:
                    if ttft is None:
                        ttft = now - start
                    prev_text_len = len(text)
                final_text = text
        end = time.monotonic()
        if ttft is None:
            return {"success": False, "error": "no tokens", "index": req_idx}
        generated = final_text[len(prompt):] if len(final_text) > len(prompt) else ""
        output_tokens = count_tokens(generated) if generated else 0
        return {"success": True, "ttft": ttft, "total_time": end - start,
                "output_tokens": output_tokens, "index": req_idx}
    except Exception as exc:
        return {"success": False, "error": str(exc), "index": req_idx}


async def bench_one_vllm(session: aiohttp.ClientSession, url: str,
                         prompt: str, max_tokens: int, model: str, req_idx: int) -> dict:
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0, "stream": True}
    start = time.monotonic()
    ttft = None
    generated_parts: list[str] = []
    try:
        timeout = aiohttp.ClientTimeout(total=600)
        async with session.post(url, json=payload, timeout=timeout) as resp:
            if resp.status != 200:
                body = await resp.text()
                return {"success": False, "error": f"HTTP {resp.status}: {body[:200]}", "index": req_idx}
            async for line in _read_lines(resp):
                now = time.monotonic()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if not choices or not choices[0].get("text"):
                        continue
                except json.JSONDecodeError:
                    continue
                if ttft is None:
                    ttft = now - start
                generated_parts.append(choices[0]["text"])
        end = time.monotonic()
        if ttft is None:
            return {"success": False, "error": "no tokens", "index": req_idx}
        generated = "".join(generated_parts)
        output_tokens = count_tokens(generated) if generated else 0
        return {"success": True, "ttft": ttft, "total_time": end - start,
                "output_tokens": output_tokens, "index": req_idx}
    except Exception as exc:
        return {"success": False, "error": str(exc), "index": req_idx}


# ── Main ───────────────────────────────────────────────────────────────────

async def main():
    ap = argparse.ArgumentParser(description="Send N concurrent requests to MoLink/vLLM")
    ap.add_argument("--url", required=True, help="Endpoint URL")
    ap.add_argument("--type", required=True, choices=["molink", "vllm"])
    ap.add_argument("--model", default="/gxq/Qwen3-14B", help="Model name (vLLM only)")
    ap.add_argument("--tokenizer", default=None, help="Tokenizer path")
    ap.add_argument("--input-tokens", type=int, default=1024)
    ap.add_argument("--output-tokens", type=int, default=512)
    ap.add_argument("--concurrent", type=int, default=3, help="Number of concurrent requests")
    ap.add_argument("--output", default=None, help="JSON results file")
    args = ap.parse_args()

    get_tokenizer(args.tokenizer or args.model)

    prompts = [generate_prompt(args.input_tokens, i) for i in range(args.concurrent)]

    print(f"Sending {args.concurrent} concurrent requests to {args.url}")
    print(f"Type: {args.type}, Input tokens: {args.input_tokens}, Output tokens: {args.output_tokens}")

    connector = aiohttp.TCPConnector(limit=0, keepalive_timeout=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        bench_start = time.monotonic()
        if args.type == "molink":
            coros = [bench_one_molink(session, args.url, p, args.output_tokens, i)
                     for i, p in enumerate(prompts)]
        else:
            coros = [bench_one_vllm(session, args.url, p, args.output_tokens, args.model, i)
                     for i, p in enumerate(prompts)]
        results = await asyncio.gather(*coros)
    bench_end = time.monotonic()

    ok = [r for r in results if r["success"]]
    fail = [r for r in results if not r["success"]]

    print(f"\n{'='*60}")
    print(f"  Completed: {len(ok)}/{args.concurrent} successful")
    print(f"  Wall time: {bench_end - bench_start:.1f}s")
    if ok:
        ttfts = [r["ttft"] * 1000 for r in ok]
        print(f"  TTFT: avg {sum(ttfts)/len(ttfts):.0f}ms, min {min(ttfts):.0f}ms, max {max(ttfts):.0f}ms")
        total_times = [r["total_time"] * 1000 for r in ok]
        print(f"  Total: avg {sum(total_times)/len(total_times):.0f}ms, min {min(total_times):.0f}ms, max {max(total_times):.0f}ms")
    if fail:
        print(f"  Failed: {len(fail)}")
        for f in fail[:3]:
            print(f"    [{f['index']}]: {f['error']}")
    print(f"{'='*60}")

    output = {
        "config": {"url": args.url, "type": args.type, "concurrent": args.concurrent,
                   "input_tokens": args.input_tokens, "output_tokens": args.output_tokens},
        "results": {
            "successful": len(ok), "failed": len(fail),
            "wall_time_s": bench_end - bench_start,
            "requests": [{"index": r["index"], "success": r["success"],
                          "ttft_ms": r.get("ttft", 0) * 1000,
                          "total_ms": r.get("total_time", 0) * 1000,
                          "output_tokens": r.get("output_tokens", 0)}
                         for r in results],
        },
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2))
        print(f"Saved to {args.output}")

    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
