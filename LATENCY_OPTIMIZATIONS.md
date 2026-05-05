# Latency Optimizations

All measurements use text-only mode (`--text`), warm OpenAI API.
Baseline on branch `rag-hypothesis-retrieval`; optimizations on `latency-optimizations`.

LLM call latency varies run-to-run (API conditions, retries). Numbers below are representative,
not hard benchmarks.

---

## Baseline

**Architecture:** each `diary_response` call spawned a fresh `python -c "..."` subprocess inside
Docker via `docker compose exec`, loading the NER model from disk every single time.

| Path | Time |
|------|------|
| RAG pipeline | 40.6s |
| Ablation pipeline | 5.6s |
| **Total (sequential)** | **46.2s** |

### RAG pipeline breakdown

| Stage | Time | % |
|-------|------|---|
| Thought tracing (ToM) | 23.1s | 62% |
| — NER model load | **10.8s** | dominant |
| — Perception tracking (batch LLM) | 3.0s | |
| — Hypothesis init (batch LLM) | 6.4s | |
| — Chain aggregate (LLM) | 2.4s | |
| RAG summarization (LLM) | 4.6s | 12% |
| Coaching intent (LLM) | 3.6s | 10% |
| Final response (LLM) | 3.5s | 9% |
| RAG key gen (LLM) | 1.9s | 5% |
| Retrieval (embed + FAISS) | 0.5s | 1% |
| History load / storage | ~0s | — |
| **Total** | **37.2s** | |

**Root cause:** NER model loaded fresh from disk on every call (10.8s), even though weights never change.

---

## Optimization 1 — Persistent server + NER model cache

### What changed

Replaced `docker compose exec python -c "..."` with a persistent HTTP server (`server.py`) running
inside the container. The server stays alive between requests, so the NER model is loaded once on
first request and cached in memory for all subsequent ones via a module-level `_MODEL_CACHE` dict.

The `ThreadingMixIn` server handles concurrent requests, which is required for `--parallel` (opt 2).

**Safe for collaborators:** `docker compose up` works exactly as before. No host-side changes needed.
Port 8765 must be free.

**Files changed:**
- `server.py` *(new)* — stdlib `http.server` + `ThreadingMixIn`, listens on port 8765
- `docker-compose.yaml` — command: `python server.py`; port 8765 exposed
- `interface/diary.py` — `diary_response` POSTs JSON to `http://localhost:8765/{rag|gpt}`
- `thought_trace/extractor/inference.py` — `_MODEL_CACHE` dict; `load_model` returns cached result on repeat calls

### Results (measured, same entry, warm server)

| Entry | Before | After | Δ |
|-------|--------|-------|---|
| 1st in session (cold model) | 46.2s | ~45s | minimal — model loads once |
| 2nd+ in session (warm model) | ~46s | **~20–25s** | **−20–25s (~55% faster)** |

NER load confirmed `0.00s` on warm entries (verified in server logs). Remaining time is pure LLM
API latency across ~6 sequential calls, which varies with API conditions.

---

## Optimization 2 — Parallel RAG + ablation paths (`--parallel`)

### What changed

Added `--parallel` flag to `mvp.py`. Both `diary_response` calls (RAG and ablation) are submitted
concurrently via `ThreadPoolExecutor(max_workers=2)`. Wall time becomes
`max(t_rag, t_ablation)` instead of `t_rag + t_ablation`.

**Files changed:**
- `mvp.py` — `--parallel` argparse flag; `ThreadPoolExecutor` branch

### Results (measured, warm model)

| Mode | Wall time |
|------|-----------|
| Sequential | ~24.8s |
| `--parallel` | **~24.2s** |
| Savings | **~0.6–2s** |

Savings are modest because the ablation path is fast (~1–2s for a single LLM call). The gain
equals the ablation path time, which varies by entry. Parallel is more valuable when API latency
is higher or ablation takes longer.

### Usage

```bash
# Sequential (default) — prints per-path timing breakdown each turn
python mvp.py --text --user yourname

# Parallel — both paths run concurrently
python mvp.py --text --user yourname --parallel
```

---

## Combined result

| | Time |
|--|------|
| Baseline (cold, sequential) | 46.2s |
| Optimized (warm, sequential) | ~20–25s |
| Optimized (warm, `--parallel`) | ~20–24s |
| **Best-case improvement** | **~−22–26s (−55–60%)** |

The dominant win is opt 1 (NER cache). Opt 2 adds a small additional saving on top.
Variance across runs is driven by OpenAI API latency, not local code.

---

## What was instrumented (no behavior change)

Timing was added throughout the pipeline to identify the above bottlenecks. These prints are
active in the current branch and surface in the Docker logs (`docker compose logs app`).

- `thought_trace/tracer.py` — per-stage timing inside `_trace` and `preprocess_input`
- `interface/io_rag.py` — per-stage timing printed at end of `generate_rag_response`
- `mvp.py` — per-path timing printed after each turn
