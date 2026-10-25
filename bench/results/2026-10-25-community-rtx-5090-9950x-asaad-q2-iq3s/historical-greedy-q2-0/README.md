# Community benchmark on NVIDIA GeForce RTX 5090

Measured on 2026-10-25 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5090, 32 GiB VRAM
- CPU: AMD Ryzen 9 9950X 16-Core Processor (16 cores, 32 threads); installed RAM: 79.7 GiB; storage: not measured
- OS: Windows 11 (10.0.26300); GPU driver: 617.42; CUDA: not read
- Strata: project ?, commit not a git checkout; engine 0.1.50
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-q2_0
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 245.0 (181.5-332.2) | 0.17 (0.16-0.20) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 5900.7 (5884.3-5957.6) | not measured | 0.71 (0.70-0.71) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 7504.4 (7487.3-7537.2) | not measured | 2.17 (2.15-2.18) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 7520.9 (7480.0-7571.6) | not measured | 4.29 (4.27-4.33) |
| agent session, follow-up turns | 9007 (8982-9036) at the last turn | see below | up to 64 | 3 | see below | 250.2 (100.5-320.9) | 0.40 (0.38-0.50) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 71.0 % (70.7-71.0); earlier-turn tokens read again per session: 4 (3-73) (earlier turns that the cache did not hold; the 8207 (8207-8207) new tokens of the files are read once); first turn (nothing cached) first token: 0.69 s (0.59-0.72).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 1084.6 | 0.594 | 222.7 |
| 1 | 2269 | 628 | 1641 | 4486.1 | 0.399 | 239.6 |
| 2 | 3975 | 2332 | 1643 | 4370.8 | 0.411 | 259.0 |
| 3 | 5681 | 4038 | 1643 | 4259.8 | 0.398 | 216.7 |
| 4 | 7342 | 5700 | 1642 | 4389.2 | 0.406 | 253.7 |
| 5 | 9007 | 7365 | 1642 | 4182.4 | 0.419 | 320.9 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 40 s.
