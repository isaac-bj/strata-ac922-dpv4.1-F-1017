# Qwen3.8 Flash-Next Q2_0 and IQ3_S on RTX 5090 / Ryzen 9 9950X

Submitted by **Asaad AlSharif** ([GitHub @AsaadAlSharif](https://github.com/AsaadAlSharif), [email](mailto:Asaad_Alsharif@hotmail.com)). Fresh measurements: 25 October 2026 UTC, 26 October in Dubai. Commands were executed with AI assistance on the submitter's PC; the official runner's measurements are retained.

Two fresh standard-protocol runs use the **official Strata 0.2.5 Windows release, engine 0.1.50**. This is the tested installed release, not a claim to test the newer 0.2.6/0.1.51 release. No engine code changes are submitted. The protocol is **draft** `strata-qwen38-thinking-v1-draft.1`; these reports are not maintainer-approved results or a global ranking.

## Machine and shared configuration

- NVIDIA GeForce RTX 5090: 32,607 MiB reported VRAM, 600 W reported power limit; AMD Ryzen 9 9950X, 16 cores / 32 threads.
- 96 GiB physically installed (two 48 GiB DIMMs); Windows exposes approximately 79.7 GiB. Configured memory rate: 6000 MT/s. See [CIM notes](hardware-notes.json); the older quick runner's “installed RAM: 79.7 GiB” label reflects OS-visible memory.
- Windows 11 build 10.0.26300; NVIDIA driver 617.42; Samsung SSD 9100 PRO 4TB NVMe. The native collector reports CUDA 13.4; the downloaded engine package is the Windows CUDA 13 release. No source/toolkit build was performed here.
- Released runner/build commit `ad638b93e59e13f06b1989574d69027718413bb3`; source checkout tag v0.2.5 is `fa663fabc5c2c568d6035b3ae853b38e1762865a`. These are software revisions, not model-weight repository revisions.
- Both fresh configurations: 131,072 context, vision disabled, server capacity one, INT8 KV with 32,768 resident cells, automatic expert cache/prefill, native MTP (speculation window 4). Complete effective options, templates, file hashes and per-request accounting accompany each result.
- Track **custom**: the explicit vision/capacity/context setup choices are recorded. The official sampling and frozen fixtures were not changed: thinking/preserve-thinking enabled, xhigh effort, temperature 1, top-p 0.95, top-k 20, min-p 0, presence penalty 0, repetition penalty 1.
- Windows desktop, IIS, control service and the benchmark observer remained running. Qwen's agent portal was stopped during the measurements, PrimeNawa was unloaded, and no other inference client was used. No game or concurrent inference stress test was run.

## Fresh draft-standard results

Three repetitions per performance case. Values are **median (minimum–maximum)**. Actual prompt counts and generated-token counts, seeds, raw timings, truncated answers and excluded metrics remain in the native reports.

| Quantization | Case | Actual prompt tokens | Prompt tokens/s | Decode tokens/s | First token, seconds |
|---|---|---:|---|---|---|
| Q2_0 | short | 77 | 251.9 (243.4–254.1) | 270.7 (230.3–287.7) | 0.313 (0.313–0.344) |
| Q2_0 | prompt-4k | 4093 | 6098.0 (5960.4–6216.6) | 258.0 (243.5–281.7) | 0.703 (0.687–0.703) |
| Q2_0 | prompt-16k | 16387 | 7747.6 (7718.8–7774.1) | 220.4 (213.0–249.4) | 2.141 (2.125–2.156) |
| IQ3_S | short | 77 | 384.0 (351.0–384.8) | 219.5 (212.0–226.0) | 0.219 (0.218–0.219) |
| IQ3_S | prompt-4k | 4093 | 5797.5 (5790.9–5813.1) | 195.0 (181.6–228.1) | 0.735 (0.734–0.750) |
| IQ3_S | prompt-16k | 16387 | 7091.8 (7011.4–7092.4) | 208.9 (186.1–215.9) | 2.343 (2.343–2.375) |

| Quantization | Required request-phase seconds | Sanity checks | Performance requests reaching cap | Failed measured requests | Skipped measured requests |
|---|---:|---:|---:|---:|---:|
| Q2_0 | 39.312 | 12/12 | 5/9 | 0 | 0 |
| IQ3_S | 44.469 | 12/12 | 6/9 | 0 | 0 |

Each suite contains **21 measured requests plus one excluded warm-up**: nine performance requests and twelve sanity requests (four instruction, four retrieval, four tool-call checks). Both native reports confirm **zero server requests before the suite**. There was one fresh process per quantization, in Q2_0 then IQ3_S order; this was not an order-swapped or multi-session significance study.

The warm-up and throughput cases have protocol output caps (128/512 tokens); sanity uses 2048. Caps include reasoning and final output. A truncated performance answer is retained, not silently retried into a completed answer. Required duration includes protocol warm-up but excludes installation, loading and model hashing. Cache state evolves after the warm-up; individual cases are not cold-model measurements. The optional concurrency stage was not run.

- [Fresh Q2_0 native report](standard-q2-0/README.md), [JSON](standard-q2-0/results.json), [raw events/replies](standard-q2-0/outputs.jsonl), [sanitized engine log](standard-q2-0/engine.log).
- [Fresh IQ3_S native report](standard-iq3-s/README.md), [JSON](standard-iq3-s/results.json), [raw events/replies](standard-iq3-s/outputs.jsonl), [sanitized engine log](standard-iq3-s/engine.log).

## Earlier measurements retained separately

- [Earlier Q2_0 draft-standard run](historical-standard-q2-0/README.md): twelve sanity checks passed; its engine had already answered **one** readiness request. It is not presented as a fresh-engine run. Decode medians: 264.0 / 261.5 / 250.1 tokens/s for short / 4K / 16K. Different initial state/background conditions prevent treating the new run as a controlled speed improvement.
- [Earlier Q2_0 greedy quick suite](historical-greedy-q2-0/README.md): thinking off and temperature zero; nine short-chat requests measured 245.0 tokens/s median (181.5–332.2), a fresh 32,049-token prompt measured 7520.9 prompt tokens/s median and 4.29 s median first-token latency. Synthetic conversation follow-ups measured 250.2 decode tokens/s median and 0.40 s first-token latency. This is not the draft standard, a real coding-agent task, or a model-quality test. Raw per-request data is included.
- [Earlier browser checks before restart](historical-browser/web-chat-before-restart.json) returned the exact test text in 0.790 s and Arabic arithmetic result 391 in 0.481 s. [After restart](historical-browser/web-chat.json), the first response took **83.788 s**, followed by 0.533 s. This poor first-response observation is retained. It is a single browser observation, not a standardized cold-start distribution; it does not establish one exclusive cause.

## Reproduction and provenance

Run the unmodified released 0.2.5 Terminal with its engine 0.1.50 and native original Qwen3.8-Flash-Next Q2_0 / IQ3_S setup. Match the saved context, vision, capacity and effective options in each result's `config.json`. Native weight filenames include `GSQ-RCO`; the exact names and SHA-256 values, including the PLE shard and expert profile, are in `model.files`. No model weights or generated packs are included here.

The normal managed launcher sends a one-token readiness check. For these fresh results, that prepared instance was stopped first; the official runner then started a separate fresh engine using its **profile mode**, waited for loaded health, hashed the files, ran the frozen suite and stopped its own instance. No chat request was sent to that measured engine before the suite. Native commands (with installation-specific paths replaced) are preserved in `run.command`:

```powershell
strata --version
strata bench run --profile <native-q2-0-run-config.json> --track custom --quantization Q2_0
strata bench run --profile <native-iq3-s-run-config.json> --track custom --quantization IQ3_S
strata bench report standard-q2-0
strata bench report standard-iq3-s
```

The quantization, source-weight, tokenizer and model-card repository revisions that the runner could not read remain **null with reasons**, rather than guessed. File hashes do not establish those missing immutable repository revisions. Native template and corpus provenance artifacts are included.

The runner performed its normal sanitization. Publication copies additionally replace the local installation root with `<strata-install>`; only path strings changed. Original local reports were retained. No numeric measurements, model replies, benchmark prompts, request outcomes or hashes were edited.

## Interpretation and limits

These are throughput/latency measurements and **small sanity checks**, not a full quality, coding, reasoning or agent benchmark. Passing 12/12 checks does not establish equivalence to full-precision Qwen or prove that the quantizations have equal quality. Native CUDA default approximations, including missed-expert tail skipping, were retained; no tail-skip-off or reference-logit comparison was performed. This RTX 5090 report does not independently reproduce the RTX 5070 claim.

Not run: optional concurrency, full-context sweep, long-context needle study, HumanEval/SWE-bench, vision in the measured suites, power/energy efficiency, cross-OS/backend testing or the newer 0.2.6 release. The portal and IQ3_S with vision were restored after measurement; that operational state is separate from the vision-off benchmark configurations.
