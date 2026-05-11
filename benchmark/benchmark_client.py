#!/usr/bin/env python3
"""
Benchmark client for vLLM and MoLink performance testing.

Measures three metrics:
  - Throughput: total output tokens / benchmark wall time (tokens/s)
  - TTFT: Time To First Token, from request sent to first token received
  - TPOP: Time Per Output Token = (total_time - ttft) / (num_output_tokens - 1)

Supports streaming for both MoLink (/generate) and vLLM (/v1/completions).

Usage:
    python benchmark_client.py \
        --url http://localhost:8080/generate \
        --type molink \
        --input-tokens 1024 --output-tokens 512 \
        --rps 1 --duration 40 \
        --output result.json
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_DEFAULT_TOKENIZER_PATH = "/home/emnets-2/gxq/Qwen3-14B"
_FALLBACK_CHARS_PER_TOKEN = 6.5

_tokenizer_instance = None


def get_tokenizer(path: str | None = None):
    global _tokenizer_instance
    if _tokenizer_instance is not None:
        return _tokenizer_instance
    try:
        from transformers import AutoTokenizer
        p = path or _DEFAULT_TOKENIZER_PATH
        _tokenizer_instance = AutoTokenizer.from_pretrained(p)
        logger.info("Loaded tokenizer from %s", p)
    except Exception as exc:
        logger.warning("Failed to load tokenizer (%s), using char-ratio fallback", exc)
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


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

_PROMPT_BLOCKS = [
    "Artificial intelligence has fundamentally transformed the way we interact with "
    "technology in the modern era. From natural language processing to computer vision, "
    "machine learning models have achieved remarkable capabilities that were once thought "
    "to be exclusively human domains. The development of large language models represents "
    "a significant milestone in this journey, enabling machines to understand and generate "
    "human language with unprecedented fluency and coherence. These models leverage vast "
    "amounts of training data and sophisticated neural network architectures to capture "
    "complex patterns in language, reasoning, and knowledge.",

    "The history of computing stretches back centuries, from Charles Babbage's analytical "
    "engine to the modern quantum computers being developed today. Each generation of "
    "computing technology has brought exponential increases in processing power and "
    "efficiency. The invention of the transistor in the mid-twentieth century revolutionized "
    "electronics and laid the groundwork for the integrated circuits that power our modern "
    "world. Today, semiconductor fabrication continues to push the boundaries of what is "
    "physically possible, with transistor sizes measured in single-digit nanometers.",

    "Climate change represents one of the most significant challenges facing humanity in "
    "the twenty-first century. Rising global temperatures are driving shifts in weather "
    "patterns, rising sea levels, and increasing frequency of extreme weather events. "
    "Scientists around the world are working to develop renewable energy sources, carbon "
    "capture technologies, and sustainable agricultural practices to mitigate these effects. "
    "International cooperation through agreements like the Paris Climate Accord demonstrates "
    "the global recognition of this pressing issue and the need for coordinated action.",

    "The field of neuroscience has made remarkable strides in understanding the human brain, "
    "arguably the most complex structure in the known universe. Advanced imaging techniques "
    "such as functional magnetic resonance imaging and positron emission tomography have "
    "allowed researchers to observe brain activity in real time. These tools have revealed "
    "the intricate networks of neurons that underlie consciousness, memory, emotion, and "
    "decision-making. The emerging field of neuroplasticity has shown that the brain can "
    "reorganize itself by forming new neural connections throughout life.",

    "Modern cryptography forms the backbone of digital security, protecting everything from "
    "online banking transactions to private communications. Public key cryptography, first "
    "proposed in the nineteen seventies, enables secure communication over insecure channels "
    "without requiring a shared secret. The advent of quantum computing poses a potential "
    "threat to current cryptographic systems, spurring research into post-quantum algorithms "
    "that can resist attacks by quantum computers. Standards bodies are actively evaluating "
    "and standardizing these next-generation cryptographic primitives.",
]

_PROMPT_TEXT = " ".join(_PROMPT_BLOCKS)


@dataclass
class RequestResult:
    success: bool
    ttft: float | None = None
    total_time: float | None = None
    output_tokens: int = 0
    tpop: float | None = None
    error: str | None = None
    generated_text: str = ""


def generate_prompt(target_tokens: int) -> str:
    """Generate a prompt of exactly *target_tokens* tokens.

    Repeats diverse text blocks and truncates to the exact token count
    using the real tokenizer (or char-ratio fallback).
    """
    repeats = (target_tokens * int(_FALLBACK_CHARS_PER_TOKEN) * 2) // len(_PROMPT_TEXT) + 1
    long_text = (_PROMPT_TEXT + " ") * repeats
    return truncate_to_tokens(long_text, target_tokens)


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------

async def _read_lines(response):
    """Yield decoded, non-empty lines from a streaming aiohttp response."""
    buf = b""
    async for chunk in response.content.iter_any():
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            decoded = line.decode("utf-8").strip()
            if decoded:
                yield decoded


# ---------------------------------------------------------------------------
# Per-request benchmarks
# ---------------------------------------------------------------------------

async def _bench_molink(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    max_tokens: int,
) -> RequestResult:
    """Benchmark a single MoLink streaming /generate request."""
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    start = time.monotonic()
    ttft = None
    prev_text_len = len(prompt)
    final_text = ""

    try:
        timeout = aiohttp.ClientTimeout(total=600)
        async with session.post(url, json=payload, timeout=timeout) as resp:
            if resp.status != 200:
                body = await resp.text()
                return RequestResult(False, error=f"HTTP {resp.status}: {body[:200]}")

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
        total_time = end - start
        if ttft is None:
            return RequestResult(False, error="no tokens received")

        generated = final_text[len(prompt):] if len(final_text) > len(prompt) else ""
        output_tokens = count_tokens(generated) if generated else 0
        tpop = (total_time - ttft) / output_tokens if output_tokens > 0 else 0.0
        return RequestResult(True, ttft=ttft, total_time=total_time,
                             output_tokens=output_tokens, tpop=tpop,
                             generated_text=generated)
    except Exception as exc:
        return RequestResult(False, error=str(exc))


async def _bench_vllm(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    max_tokens: int,
    model: str,
) -> RequestResult:
    """Benchmark a single vLLM streaming /v1/completions request."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    start = time.monotonic()
    ttft = None
    generated_parts: list[str] = []

    try:
        timeout = aiohttp.ClientTimeout(total=600)
        async with session.post(url, json=payload, timeout=timeout) as resp:
            if resp.status != 200:
                body = await resp.text()
                return RequestResult(False, error=f"HTTP {resp.status}: {body[:200]}")

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
        total_time = end - start
        if ttft is None:
            return RequestResult(False, error="no tokens received")

        generated = "".join(generated_parts)
        output_tokens = count_tokens(generated) if generated else 0
        tpop = (total_time - ttft) / output_tokens if output_tokens > 0 else 0.0
        return RequestResult(True, ttft=ttft, total_time=total_time,
                             output_tokens=output_tokens, tpop=tpop,
                             generated_text=generated)
    except Exception as exc:
        return RequestResult(False, error=str(exc))


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _pct(data: list[float], p: float) -> float | None:
    if not data:
        return None
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] * (c - k) + s[c] * (k - f)


def _stats(data: list[float]) -> dict[str, float | None]:
    if not data:
        return {k: None for k in ("avg", "p50", "p90", "p99", "min", "max")}
    return {
        "avg": round(sum(data) / len(data), 6),
        "p50": round(_pct(data, 50), 6),
        "p90": round(_pct(data, 90), 6),
        "p99": round(_pct(data, 99), 6),
        "min": round(min(data), 6),
        "max": round(max(data), 6),
    }


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

async def run_benchmark(
    url: str,
    endpoint_type: str,
    prompt: str,
    max_tokens: int,
    rps: float,
    duration: int,
    model: str,
) -> dict[str, Any]:
    """Launch requests at *rps* for *duration* seconds and collect metrics."""

    total_requests = int(rps * duration)
    interval = 1.0 / rps

    connector = aiohttp.TCPConnector(limit=0, keepalive_timeout=0)
    tasks: list[asyncio.Task] = []

    async with aiohttp.ClientSession(connector=connector) as session:
        bench_start = time.monotonic()
        for i in range(total_requests):
            if endpoint_type == "molink":
                coro = _bench_molink(session, url, prompt, max_tokens)
            else:
                coro = _bench_vllm(session, url, prompt, max_tokens, model)
            tasks.append(asyncio.create_task(coro))
            if i < total_requests - 1:
                await asyncio.sleep(interval)

        results: list[RequestResult] = await asyncio.gather(*tasks)
    bench_end = time.monotonic()
    bench_duration = bench_end - bench_start

    ok = [r for r in results if r.success]
    fail = [r for r in results if not r.success]

    ttfts = [r.ttft for r in ok if r.ttft is not None]
    tpops = [r.tpop for r in ok if r.tpop is not None]
    total_tokens = sum(r.output_tokens for r in ok)

    summary: dict[str, Any] = {
        "config": {
            "url": url,
            "type": endpoint_type,
            "input_tokens": count_tokens(prompt),
            "max_tokens": max_tokens,
            "rps": rps,
            "duration_s": duration,
            "total_requests": total_requests,
        },
        "results": {
            "successful_requests": len(ok),
            "failed_requests": len(fail),
            "total_output_tokens": total_tokens,
            "benchmark_wall_time_s": round(bench_duration, 3),
            "throughput_tokens_per_s": round(total_tokens / bench_duration, 3)
                                       if bench_duration > 0 else 0,
            "ttft_s": _stats(ttfts),
            "tpop_s": _stats(tpops),
            "request_latency_s": _stats([r.total_time for r in ok
                                         if r.total_time is not None]),
        },
    }
    if fail:
        summary["errors"] = [{"request_id": i, "error": r.error}
                             for i, r in enumerate(results) if not r.success][:10]
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Benchmark client for vLLM / MoLink")
    ap.add_argument("--url", required=True, help="Endpoint URL")
    ap.add_argument("--type", required=True, choices=["molink", "vllm"])
    ap.add_argument("--input-tokens", type=int, default=1024)
    ap.add_argument("--output-tokens", type=int, default=512)
    ap.add_argument("--rps", type=float, required=True, help="Requests per second")
    ap.add_argument("--duration", type=int, default=40, help="Benchmark duration (s)")
    ap.add_argument("--output", default=None, help="Path to write JSON results")
    ap.add_argument("--model", default="/gxq/Qwen3-14B",
                    help="Model name for vLLM /v1/completions (ignored for molink)")
    ap.add_argument("--tokenizer", default=None,
                    help="Tokenizer path (default: auto-detect from --model)")
    ap.add_argument("--prompt-file", default=None,
                    help="Read prompt from file instead of generating one")
    args = ap.parse_args()

    # Initialize tokenizer
    get_tokenizer(args.tokenizer or args.model)

    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text()
    else:
        prompt = generate_prompt(args.input_tokens)

    actual_input_tokens = count_tokens(prompt)
    print(f"Prompt: {actual_input_tokens} tokens ({len(prompt)} chars)")
    print(f"Config: type={args.type}  rps={args.rps}  duration={args.duration}s  "
          f"max_tokens={args.output_tokens}")
    print(f"Target: {args.url}")

    summary = asyncio.run(run_benchmark(
        url=args.url,
        endpoint_type=args.type,
        prompt=prompt,
        max_tokens=args.output_tokens,
        rps=args.rps,
        duration=args.duration,
        model=args.model,
    ))

    r = summary["results"]
    print()
    print("=" * 60)
    print(f"  Success: {r['successful_requests']}/{summary['config']['total_requests']}")
    print(f"  Throughput : {r['throughput_tokens_per_s']:.1f} tokens/s")
    ttft = r["ttft_s"]
    if ttft["avg"] is not None:
        print(f"  TTFT       : avg {ttft['avg']*1000:.1f}ms  "
              f"p50 {ttft['p50']*1000:.1f}ms  p99 {ttft['p99']*1000:.1f}ms")
    tpop = r["tpop_s"]
    if tpop["avg"] is not None:
        print(f"  TPOP       : avg {tpop['avg']*1000:.2f}ms  "
              f"p50 {tpop['p50']*1000:.2f}ms  p99 {tpop['p99']*1000:.2f}ms")
    print("=" * 60)

    out = json.dumps(summary, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(out)
        print(f"Saved to {args.output}")
    else:
        print(out)


if __name__ == "__main__":
    main()
