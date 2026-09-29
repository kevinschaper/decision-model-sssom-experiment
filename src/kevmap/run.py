"""POST every eval item to a running kev.serve; append raw answers to results/<model>/responses.jsonl (resumable)."""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from .items import load_items
from .paths import RESULTS
from .questions import build_request


def result_dir(model: str, items: str = "items"):
    return RESULTS / model / items

MAX_SERVER_GB = float(os.environ.get("KEVMAP_MAX_SERVER_GB", "18"))
REL_TOPN = int(os.environ.get("KEVMAP_REL_TOPN", "3"))  # match-only items: relation asked for the top-N by p(match)
SPLIT = os.environ.get("KEVMAP_SPLIT_QUESTIONS", "0") == "1"  # one question per request (Hopper's server requires it)
API_KEY = os.environ.get("KEVMAP_API_KEY")  # bearer token for a hosted endpoint (Jev); read from the environment only
MAX_INPUT_TOKENS = int(float(os.environ.get("KEVMAP_MAX_INPUT_TOKENS", "0")))  # hard stop for a paid endpoint; 0 = off
USD_PER_M_INPUT = float(os.environ.get("KEVMAP_USD_PER_M_INPUT", "0.042"))  # Jev list price, output is free


def _post_once(client: httpx.Client, req: dict) -> dict:
    """One POST with retries on rate limits and transient server errors (a hosted endpoint throttles)."""
    for attempt in range(6):
        r = client.post("/v1/systemone", json=req)
        if r.status_code in (429, 500, 502, 503, 504) and attempt < 5:
            time.sleep(min(2**attempt, 30))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("unreachable")


def post(client: httpx.Client, req: dict) -> dict:
    """POST a request; with KEVMAP_SPLIT_QUESTIONS=1 each question goes in its own request, answers merged."""
    if not SPLIT or len(req["questions"]) == 1:
        return _post_once(client, req)
    answers, latency, usage = {}, 0.0, {"input_tokens": 0, "output_tokens": 0}
    for qid, q in req["questions"].items():
        body = _post_once(client, {**req, "questions": {qid: q}})
        answers.update(body["answers"])
        latency += body.get("latency_ms") or 0.0
        for k in usage:
            usage[k] += (body.get("usage") or {}).get(k, 0)
    return {"answers": answers, "latency_ms": latency, "usage": usage}


def server_rss_gb() -> float:
    """Memory footprint of the kev server (top's MEM column: unlike ps rss it includes MLX's Metal
    allocations), so a leaking backend aborts the run, not the machine."""
    try:
        pids = subprocess.run(["pgrep", "-f", "kev.serve|serve_kev"], capture_output=True, text=True).stdout.split()
        best = 0.0
        for pid in pids:
            cmd = ["top", "-l", "1", "-pid", pid, "-stats", "mem"]
            out = subprocess.run(cmd, capture_output=True, text=True).stdout
            val = out.strip().splitlines()[-1].strip() if out.strip() else "0"
            unit = {"K": 1 / (1 << 20), "M": 1 / 1024, "G": 1.0}.get(val[-1], 0)
            best = max(best, float(val[:-1].rstrip("+-")) * unit) if unit else best
        return best
    except (OSError, ValueError, IndexError):
        return 0.0


def run(model: str, port: int, limit: int | None = None, workers: int = 1, tiers: list[str] | None = None,
        items_name: str = "items", base_url: str | None = None, request_model: str = "kev-latest") -> None:
    out_dir = result_dir(model, items_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "responses.jsonl"
    done = {json.loads(line)["item_id"] for line in out.open()} if out.exists() else set()
    items = [it for it in load_items(items_name) if it["item_id"] not in done and (not tiers or it["tier"] in tiers)]
    if limit:
        items = items[:limit]
    base = base_url or f"http://127.0.0.1:{port}"
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    spent = {"input_tokens": 0}
    with httpx.Client(base_url=base, timeout=600, headers=headers) as client:
        try:  # Hopper's server has no GET /v1/models (HTML 501); the run must not depend on it
            r = client.get("/v1/models")
            is_json = r.headers.get("content-type", "").startswith("application/json")
            info = r.json() if is_json else {"status": r.status_code}
        except (httpx.HTTPError, ValueError) as e:
            info = {"error": str(e)}
        (out_dir / "server.json").write_text(json.dumps(info, indent=2))
        print(f"{len(done)} done, {len(items)} to go -> {out}", file=sys.stderr)

        def one(item: dict) -> dict:
            t0 = time.perf_counter()
            body = post(client, build_request(item, model=request_model))
            answers = body["answers"]
            if item.get("match_only") and REL_TOPN:
                # second stage: the predicate for the candidates the model rated highest (Kev is single-pass,
                # so it cannot condition on its own match answer inside one request)
                probs = answers["match"]["probabilities"]
                top = sorted((c for c in item["candidates"]), key=lambda c: -probs.get(c["option"], 0))[:REL_TOPN]
                body2 = post(client, build_request(item, model=request_model, relation_for=top))
                answers = {**answers, **{k: v for k, v in body2["answers"].items() if k.startswith("rel_")}}
            return {
                "item_id": item["item_id"],
                "tier": item["tier"],
                "answers": answers,
                "usage": body.get("usage"),
                "latency_ms": body.get("latency_ms"),
                "wall_ms": round((time.perf_counter() - t0) * 1000),
            }

        t_start = time.perf_counter()
        with out.open("a") as f, ThreadPoolExecutor(workers) as pool:
            for n, rec in enumerate(pool.map(one, items), 1):
                f.write(json.dumps(rec) + "\n")
                f.flush()
                spent["input_tokens"] += (rec.get("usage") or {}).get("input_tokens", 0) or 0
                if n % 25 == 0 or n == len(items):
                    rate = n / (time.perf_counter() - t_start)
                    gb = server_rss_gb()
                    usd = spent["input_tokens"] / 1e6 * USD_PER_M_INPUT
                    print(f"  {n}/{len(items)}  {rate:.2f} items/s  server {gb:.1f} GB  "
                          f"input tokens {spent['input_tokens']:,} (~${usd:.3f})", file=sys.stderr)
                if MAX_INPUT_TOKENS and spent["input_tokens"] > MAX_INPUT_TOKENS:
                    print(f"input-token budget {MAX_INPUT_TOKENS:,} exceeded; stopping (resumable)", file=sys.stderr)
                    pool.shutdown(wait=False, cancel_futures=True)
                    sys.exit(4)
                    if gb > MAX_SERVER_GB:
                        print(f"server over {MAX_SERVER_GB} GB; aborting (KEVMAP_MAX_SERVER_GB)", file=sys.stderr)
                        pool.shutdown(wait=False, cancel_futures=True)
                        sys.exit(3)
