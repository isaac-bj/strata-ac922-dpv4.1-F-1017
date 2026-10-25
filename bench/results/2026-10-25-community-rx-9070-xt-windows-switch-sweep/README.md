# Community benchmark on AMD Radeon RX 9070 XT, Windows 11: gfx12 prompt switches, prompt chunk and VRAM reserve, IQ3_XXS at 131K and 262K

Measured on 2026-10-25 by DE3000, engine 0.1.48 (the ready-made Windows `rx9070` package). 92 fresh engine starts
in three sessions: 16 one-change arms and 7 combinations at a 131,072-token context, 8 arms at 262,144 tokens with
prompts up to 256,326 tokens, the method of
[2026-10-10-community-rx-9070](../2026-10-10-community-rx-9070/README.md) repeated twice, and the two test pictures
through both picture readers.

What it answers: `docs/AMD_HIP.md` keeps the matrix-core prompt attention (`STRATA_HIP_WMMA`) opt-in on Windows
"until tested on Windows". On this card it read prompts 9-13% faster on its own, and 12-21% faster with
`STRATA_PF_FUSED=1` and `STRATA_SELECT_WMMA=1`, with decode unchanged. Combined with larger prompt chunks
(`STRATA_PREFILL_AUTO_MAX=16384`) a 256K prompt read at 2,057-2,096 tok/s against 968 tok/s with the base settings.
Main limitation: one machine, other programs held 2.4-2.6 GiB of VRAM throughout, and no answer-quality check beyond
the recall and picture checks (the standard benchmark runs of the same settings are in a separate pull request).

## Hardware and software

- **GPU:** AMD Radeon RX 9070 XT 16 GB (gfx1201), driver 32.0.32015.7029; Adrenalin tuning on Default, Smart Access
  Memory (Resizable BAR) on; Windows budgets 15,419 of 16,304 MiB to the engine. Clocks, power and temperature not
  recorded.
- **CPU / RAM:** AMD Ryzen 7 9850X3D (8 cores, 16 threads); 63.6 GiB installed RAM, speed not recorded.
- **Storage:** WD_BLACK SN850X HS 2000GB (NVMe), holding the model files and the pack.
- **PCIe:** Gen5 x16, current and maximum, as Windows reports the card's link (`DEVPKEY_PciDevice_CurrentLinkSpeed` /
  `Width`, read after the runs); the engine's start-up probe read 54.0-54.8 GB/s host to device over all starts.
- **OS:** Windows 11, build 10.0.26300.
- **Strata:** commit `7dd5cf18` (v0.2.4.2), source checkout; engine 0.1.48, the prebuilt package (`BUILD.json`: commit
  `6f8800aefe27`, ROCm 10.2.0a20260930, hipBLASLt 100500, archs gfx1200 / gfx1201).
- **Background:** the desktop in normal use; other programs held 2,498-2,686 MiB of VRAM at the start of every engine
  start (per start in `data/starts.json`), 48-55 GiB RAM free.

## Model and configuration

- Qwen3.8-Flash-Next GSQ-RCO **IQ3_XXS** (ISTA-DASLab, downloaded by setup; the hashes below are the ones `setup.py`
  pins for revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09` of `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`):
  `...-00001-of-00002.gguf` SHA-256 `219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15`,
  `...-00002-of-00002.gguf` SHA-256 `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` (the same files
  as the precedent report). Pack built by setup; expert profile `data/expert-profile.bin` SHA-256
  `8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf`; MTP draft head built by setup.
- **Base configuration** ([configs/run-sweep-base.json](configs/run-sweep-base.json)): the precedent's settings on
  this install's files. No calibration, no experimental speed projection, no vision unless stated.

  ```text
  --pack <data>\packs\iq3_xxs --native ...IQ3_XXS-00001-of-00002.gguf --ple-gguf ...IQ3_XXS-00002-of-00002.gguf
  --expert-profile <strata>\data\expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5
  --mtp <data>\mtp\rt --max-context 131072 --kv int8 --kv-resident 32768
  env STRATA_HIPBLASLT_TUNING=<strata>\engine\tuning\gfx1201-hipblaslt-100500.txt
  ```

- The other configs change one thing each: `--max-context 262144` (`256k`), `--vram-reserve-mib 1280` / `1472`, no
  `--kv-resident` (`kvvram`), `--spec-min-p 0.70` (`minp070`), no tuning-table variable (`notable`), `--vision` with
  the CPU or GPU picture reader, and `256k-mine` (`--prefill auto:16384 --vram-reserve-mib 1280` at 262K: the config I
  used day to day before this sweep). All are in [configs/](configs/). Environment switches were set per start
  (column "Switches").

## Method

- **Harness:** the precedent's [benchmark.py](../2026-10-10-community-rx-9070/benchmark.py), unchanged: synthetic
  Python-like filler sized by a calibration request, a fresh nonce in every prompt (every measured request read
  `0 reused` tokens), temperature 0, `reasoning_effort: none`, streaming, 256 tokens out (32 for recall checks). The
  calibration request and a warm-up request (2,082 tokens) are discarded. Prompt and decode tok/s are the engine's own log
  lines, paired to each request by prompt-token count (`pairing_ok` true for every measured row).
- **Starts:** every row of every table is a **fresh server and engine** (`serve/server.py` on port 8091), stopped
  after its requests, with no saved tuning applied (`STRATA_NO_PRESET=1`). The switches were set in the environment
  of that one start only (`data/starts.json` records them per start). How the engine log of each start
  (`data/engine-logs/`) shows them: `STRATA_HIP_WMMA` with its own line (`STRATA_HIP_WMMA: the prompt attention on
  matrix cores on (gfx1201)`, or `chosen per call` for `auto`); `STRATA_PREFILL_AUTO_MAX` in the chunk size and the
  #1630 warning; `STRATA_PF_FUSED=1` in a 512-slot ring when WMMA is off (with WMMA the ring is 96 either way);
  `STRATA_GFX12_DEFAULTS=0`, `STRATA_HC_Q8=0` and `STRATA_ROUTE_TAIL_SKIP=0` by the default-on lines they remove.
  `STRATA_SELECT_WMMA=1` prints nothing.
- **Sizes:** 131K arms: 4K and 32K prompts, 3 runs each per start. 262K arms: 4K, 32K, 128K and 256K, 3 runs each.
  The 256K label is 248,000 because this filler comes out 3.4% over its label at that size; actual prompts were
  256,326 tokens (256,371 for the recall checks). In a test start before the runs, the label 256,000 made a
  262,413-token prompt, which the server refused with HTTP 400 (no room left in the context).
- **Order and noise:** each set of arms ran in three (131K) or two to three (262K) rounds, in a different order each
  round. `base` and `base2` are the same configuration and show the spread between starts; each later session
  re-measured its own reference (`base_c`, `base256`, `ref256`).
- **Sessions:** S1 2026-10-25 00:18-03:34 (one-change arms, precedent method, pictures), S2 10:03-14:01
  (combinations, 262K arms), S3 15:09-17:06 (262K follow-up). One planned start was missing from each of S1 and S2
  (see Failures); each was run in the next session.
- **Statistics:** per start the median of its 3 runs; per arm the median of the start medians, with the lowest and
  highest start median in brackets; rounded half up. Percentages compare arm medians with the session's reference:
  in the one-change table the median of the six `base` and `base2` start medians (4K 782.15, 32K 933.05 tok/s), in
  the others `base_c`, `base256` and `ref256`.

## Results

### The precedent's method (S1, one start each)

4K / 32K / 128K prompts (actual 4,195-4,199 / 33,467 / 129,973 tokens), 3 runs each, plus six recall checks.

| Configuration | Prompt tok/s 4K / 32K / 128K | Decode tok/s 4K / 32K / 128K | Recall |
| --- | --- | --- | --- |
| [2026-10-10 RX 9070](../2026-10-10-community-rx-9070/README.md), Ryzen 7 5800X3D, engine 0.1.41 | 586.2 / 804.5 / 832.9 | 58.7 / 57.9 / 57.9 | 6/6 |
| this card, base | 781.6 / 929.7 / 966.4 | 89.4 / 86.1 / 83.6 | 6/6 |
| this card, `STRATA_HIP_WMMA=1 STRATA_PF_FUSED=1` | 876.1 / 1,108.3 / 1,262.0 | 88.4 / 94.4 / 83.9 | 6/6 |

The card, CPU and engine differ from the precedent; this isolates none of them.

### One change at a time, 131,072 context (S1, 3 starts per arm)

| Arm | Switches | Config | Starts | Prompt 4K | Prompt 32K | Decode 4K | Decode 32K | Slots | Chunk / ring |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `base` | - | base | 3/3 | 782 (782–782) | 933 (933–935) | 89.4 (86.6–91.4) | 88.4 (87.1–89.8) | 3778, 3780 | 6144 / 384 |
| `base2` | - | base | 3/3 | 784 (781–785) | 932 (930–933) | 92.6 (89.7–95.5) | 87.3 (85.5–91.9) | 3778 | 6144 / 384 |
| `wmma` | `STRATA_HIP_WMMA=1` | base | 3/3 | 849 (848–853) +9% | 1,057 (1,054–1,060) +13% | 90.8 (89.4–92.2) | 90.2 (88.4–91.9) | 3778, 3780 | 6144 / 96 |
| `wmma_auto` | `STRATA_HIP_WMMA=auto` | base | 3/3 | 849 (848–852) +9% | 1,059 (1,056–1,059) +13% | 91.0 (83.7–91.8) | 88.8 (87.0–89.6) | 3778, 3780 | 6144 / 96 |
| `fused` | `STRATA_PF_FUSED=1` | base | 3/3 | 800 (798–802) +2% | 966 (966–967) +3% | 87.1 (84.8–90.3) | 90.3 (89.2–90.6) | 3778, 3780 | 6144 / 512 |
| `linux` | `STRATA_HIP_WMMA=1`, `STRATA_PF_FUSED=1` | base | 3/3 | 879 (877–880) +12% | 1,107 (1,107–1,109) +19% | 89.4 (85.6–91.4) | 88.3 (86.7–90.0) | 3778, 3780 | 6144 / 96 |
| `linux_select` | `STRATA_HIP_WMMA=1`, `STRATA_PF_FUSED=1`, `STRATA_SELECT_WMMA=1` | base | 3/3 | 879 (878–884) +12% | 1,126 (1,124–1,127) +21% | 91.5 (86.4–100.5) | 91.1 (88.7–91.4) | 3778, 3780 | 6144 / 96 |
| `gfx12_off` | `STRATA_GFX12_DEFAULTS=0` | base | 3/3 | 777 (777–777) -1% | 926 (925–928) -1% | 85.2 (79.4–86.8) | 84.2 (83.4–86.5) | 3532, 3534 | 6144 / 384 |
| `hcq8_off` | `STRATA_HC_Q8=0` | base | 3/3 | 777 (777–778) -1% | 927 (925–927) -1% | 87.9 (82.3–88.4) | 87.2 (87.0–90.4) | 3532, 3534 | 6144 / 384 |
| `tailskip_off` | `STRATA_ROUTE_TAIL_SKIP=0` | base | 3/3 | 781 (779–783) 0% | 933 (932–934) 0% | 84.9 (83.5–87.0) | 80.3 (79.9–81.2) | 3778 | 6144 / 384 |
| `prefill_8k` | `STRATA_PREFILL_AUTO_MAX=8192` | base | 2/2 * | 784 (782–785) 0% | 1,078 (1,078–1,078) +16% | 95.2 (94.0–96.3) | 87.1 (86.0–88.1) | 3778 | 8192 / 384 |
| `prefill_16k` | `STRATA_PREFILL_AUTO_MAX=16384` | base | 3/3 | 783 (781–787) 0% | 1,369 (1,362–1,369) +47% | 89.4 (89.4–91.7) | 89.0 (85.4–92.0) | 3778 | 11776 / 384 |
| `no_table` | - | notable | 3/3 | 784 (784–784) 0% | 933 (932–933) 0% | 85.3 (84.4–88.7) | 84.5 (79.8–90.2) | 3778 | 6144 / 384 |
| `reserve_1280` | - | reserve1280 | 3/3 | 1,063 (1,060–1,064) +36% | 1,238 (1,237–1,238) +33% | 94.0 (88.3–94.4) | 88.1 (86.7–89.3) | 4560 | 6144 / 384 |
| `kv_vram` | - | kvvram | 3/3 | 763 (762–765) -3% | 917 (915–917) -2% | 87.9 (82.4–88.8) | 82.9 (78.3–87.0) | 2924 | 6144 / 384 |
| `minp_070` | - | minp070 | 3/3 | 781 (780–785) 0% | 933 (932–934) 0% | 93.2 (91.5–94.3) | 90.6 (86.9–90.7) | 3778 | 6144 / 384 |

\* `prefill_8k`: two starts in S1; the third ran at the start of S2 (see Failures): prompt 4K 790 and 32K 1,082,
against 782 / 785 and 1,078 / 1,078 in S1. The table counts the S1 starts only.

Prompt speed moved by less than 0.7% between starts of the same arm here, so the prompt columns separate well. Decode
moved by up to 16% between starts of one arm; only `tailskip_off` at 32K lies entirely below every `base` / `base2`
start.

### Combinations, 131,072 context (S2, 3 starts per arm)

| Arm | Switches | Config | Starts | Prompt 4K | Prompt 32K | Decode 4K | Decode 32K | Slots | Chunk / ring |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `base_c` | - | base | 3/3 | 784 (779–787) | 934 (931–934) | 89.3 (86.3–91.2) | 85.1 (83.9–89.4) | 3735, 3742 | 6144 / 384 |
| `ls_c` | `STRATA_HIP_WMMA=1`, `STRATA_PF_FUSED=1`, `STRATA_SELECT_WMMA=1` | base | 3/3 | 880 (872–885) +12% | 1,125 (1,116–1,129) +20% | 89.5 (81.3–91.7) | 85.2 (78.6–91.6) | 3735, 3742 | 6144 / 96 |
| `c_16k` | as `ls_c` + `STRATA_PREFILL_AUTO_MAX=16384` | base | 3/3 | 878 (878–879) +12% | 1,845 (1,839–1,848) +98% | 94.3 (90.1–95.0) | 85.6 (84.9–86.8) | 3735, 3742 | 16384 / 96 |
| `c_1280` | as `ls_c` | reserve1280 | 3/3 | 906 (902–907) +16% | 1,160 (1,156–1,162) +24% | 91.9 (87.6–97.8) | 88.8 (86.8–90.5) | 4516, 4524 | 6144 / 96 |
| `c_16k_1280` | as `c_16k` | reserve1280 | 3/3 | 908 (905–908) +16% | 1,882 (1,882–1,883) +102% | 97.7 (89.7–98.5) | 87.6 (86.6–97.0) | 4516, 4532 | 16384 / 96 |
| `c_16k_1472` | as `c_16k` | reserve1472 | 3/3 † | 904 (901–905) +15% | 1,879 (1,876–1,930) +101% | 90.1 (89.4–93.7) | 92.2 (91.0–92.2) | 4403, 4417 | 16384 / 96 |
| `p16k_1280` | `STRATA_PREFILL_AUTO_MAX=16384` | reserve1280 | 3/3 | 805 (804–806) +3% | 1,342 (1,342–1,344) +44% | 88.5 (87.0–98.3) | 90.4 (85.6–92.3) | 4516 | 14848 / 384 |

† `c_16k_1472`: two starts in S2, the third (skipped by the batch file in S2) in S3.

### 262,144 context, prompts up to 256K (S2, 2 starts per arm)

| Arm | Switches | Config | Starts | Prompt 4K | Prompt 32K | Prompt 128K | Prompt 256K | Decode 4K | Decode 32K | Decode 128K | Decode 256K | Slots | Chunk / ring |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `base256` | - | 256k | 2/2 | 775 (774–776) | 926 (924–927) | 978 (978–979) | 968 (968–969) | 88.6 (86.8–90.4) | 85.0 (84.9–85.1) | 82.1 (79.4–84.8) | 80.3 (78.5–82.1) | 3602 | 6144 / 384 |
| `ls256` | `STRATA_HIP_WMMA=1`, `STRATA_PF_FUSED=1`, `STRATA_SELECT_WMMA=1` | 256k | 2/2 | 871 (869–872) +12% | 1,115 (1,115–1,116) +20% | 1,188 (1,188–1,188) +21% | 1,165 (1,164–1,166) +20% | 90.4 (88.9–91.8) | 86.7 (85.7–87.6) | 84.4 (81.7–87.0) | 80.4 (76.8–83.9) | 3581, 3602 | 6144 / 96 |
| `c256_16k_1280` | as `ls256` + `STRATA_PREFILL_AUTO_MAX=16384` | 256k-reserve1280 | 2/2 | 898 (896–900) +16% | 1,869 (1,867–1,872) +102% | 2,084 (2,083–2,086) +113% | 2,096 (2,095–2,096) +116% | 91.4 (90.0–92.8) | 87.1 (86.8–87.4) | 86.6 (86.5–86.7) | 81.7 (81.3–82.0) | 4378 | 16384 / 96 |
| `c256_16k_1472` | as `c256_16k_1280` | 256k-reserve1472 | 2/2 | 892 (891–893) +15% | 1,862 (1,861–1,862) +101% | 2,079 (2,075–2,082) +112% | 2,088 (2,088–2,089) +116% | 92.9 (92.1–93.6) | 89.6 (86.5–92.7) | 87.6 (86.2–88.9) | 81.3 (80.0–82.5) | 4264 | 16384 / 96 |
| `c256_32k_1472` | as `ls256` + `STRATA_PREFILL_AUTO_MAX=32768` | 256k-reserve1472 | 2/2 | 892 (892–893) +15% | 2,267 (2,263–2,272) +145% | 2,225 (2,223–2,228) +127% | 2,080 (2,077–2,082) +115% | 92.9 (92.0–93.7) | 87.3 (86.7–87.8) | 86.0 (84.3–87.7) | 84.3 (80.9–87.7) | 4278 | 16896 / 96 |

Recall at 256K (one run per depth, prompt 256,371 tokens): `base256` and `c256_16k_1472` found the code at 10%, 50%
and 90% (6/6). Time to first token for the 256K prompt: about 265 s with `base256`, about 123 s with
`c256_16k_1472`.

### 262,144 context, the reserve at its default and my own config (S3, 3 starts per arm)

`ref256` is `c256_16k_1280` again, as this session's reference.

| Arm | Switches | Config | Starts | Prompt 4K | Prompt 32K | Prompt 128K | Prompt 256K | Decode 4K | Decode 32K | Decode 128K | Decode 256K | Slots | Chunk / ring |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ref256` | `STRATA_HIP_WMMA=1`, `STRATA_PF_FUSED=1`, `STRATA_PREFILL_AUTO_MAX=16384`, `STRATA_SELECT_WMMA=1` | 256k-reserve1280 | 3/3 | 896 (895–896) | 1,870 (1,867–1,874) | 2,081 (2,080–2,081) | 2,096 (2,092–2,097) | 91.6 (87.0–95.9) | 87.7 (87.6–88.2) | 86.7 (82.7–89.4) | 83.6 (82.0–85.0) | 4377, 4378 | 16384 / 96 |
| `c256_16k` | as `ref256` | 256k | 3/3 | 870 (867–993) -3% | 1,830 (1,819–1,886) -2% | 2,041 (2,041–2,044) -2% | 2,057 (2,054–2,061) -2% | 86.5 (82.8–91.3) | 83.1 (83.1–89.3) | 83.1 (82.0–88.6) | 77.8 (77.3–81.9) | 3602, 3614 | 16384 / 96 |
| `mine256` | `STRATA_HIP_WMMA=1` | 256k-mine | 3/3 | 869 (867–871) -3% | 1,650 (1,643–1,652) -12% | 1,799 (1,798–1,801) -14% | 1,761 (1,756–1,762) -16% | 89.1 (88.1–92.5) | 87.2 (86.9–91.6) | 88.4 (84.1–88.5) | 81.6 (81.5–84.1) | 4378, 4394 | 13824 / 96, 14080 / 96 |

`c256_16k` 4K: one start read 850, 993 and 994 tok/s in its three runs; the other two starts read 867-874.

### Pictures (S1; `docs/TEST_REQUESTS.md`: Windows with an AMD card, a picture sent)

`tools/qwen35/pictures/shapes.png` and `code4217.png`, two questions each (the questions of
`tools/qwen35/q36_vision_serve.py`), then a text question (17 x 23), temperature 0, thinking off. Answers in
[data/pictures.json](data/pictures.json).

| Reader | Right | Answers | Seconds per picture request |
| --- | ---: | --- | --- |
| CPU (`strata-vision`, 8 threads) | 5/5 | `STRATA 42`; red circle, blue square, green triangle; `CODE 4217`; red square left, blue circle right; `391` | 2.7-5.5 |
| GPU (`strata-vision-hip`) | 5/5 | the same content; the last picture answer is worded as a list | 2.8-3.4 |

## What the engine logs show

- **The fused int8 prompt kernels run in every arm.** Every benchmark start logs `prompt experts on the fused int8
  kernels`, `base` included: for IQ3_XXS they are on by default on gfx12 on Windows too (`src/prefill/moe_fused_iq.cu`;
  only the Q2_0 default is Linux-only), while `docs/AMD_HIP.md` calls `STRATA_PF_FUSED` opt-in and says Windows was
  not measured. `STRATA_PF_FUSED=1` set explicitly still changed something: a 512-slot ring instead of 384 and +2-3%
  (the `fused` arm).
- **`no_table` is not an arm.** Without `STRATA_HIPBLASLT_TUNING` the engine still logged `hipBLASLt tuning enabled
  (32 rows, gfx1201, version 100500)`: it found the package's table itself. Its numbers equal `base`.
- **The VRAM reserve:** with 1280 the engine left 843-877 MiB free after loading and logged `WARNING: only 875 MiB of
  VRAM free at the start of this request (below 1024 MiB) ... Add --vram-reserve-mib 1472` (875-886 MiB at the
  request); with 1472 it left 1,067-1,068 MiB, 982-983 MiB at the first request, and the same warning asking for 1536.
  With the default (2,560 MiB on a 16 GB card) 2,155-2,187 MiB stayed free (2,234 MiB without `--kv-resident`). The
  reserve arms' decode medians stay within the start-to-start spread of their references; paging itself was not
  measured.
- **Prompt chunks of 8K and more:** the 30 starts with `STRATA_PREFILL_AUTO_MAX` of 8192 or more and the 3 `mine256`
  starts (`--prefill auto:16384`) logged the #1630 warning (`prompt chunks of 8192 tokens or more crashed the engine
  on Windows HIP`). None of these 33 starts crashed or restarted (each start's log has one engine start), including
  256K prompts; the Windows commit left at start was 13.8-20.1 GiB. The 32768 limit gave 16,896-token chunks (the
  engine's own fit).
- **`mine256` chunks:** `--prefill auto:16384` gave 13,824-14,080-token chunks where `ref256` (`--prefill auto` with
  `STRATA_PREFILL_AUTO_MAX=16384` and the two extra switches) got 16,384. Which of the two differences causes it was not
  isolated.

## Failures and limitations

- **Failed requests:** none. Every start became ready in 16-20 s; every measured request returned an answer (611 of
  660 prompt runs ran to the 256-token cap, 49 ended earlier with `stop`; the engine timed the tokens it made).
- **Missing starts:** my own run scripts (a batch file and its wrapper), which started the runs one by one, skipped two planned starts:
  `prefill_8k` round 2 in S1 and `c_16k_1472` round 3 in S2. Nothing in Strata was involved. Each ran in the next
  session, as marked above.
- **Background load:** other programs held 2,673-2,686 MiB of VRAM at every start in S1 and 2,498-2,545 MiB in S2 and
  S3; their names are left out of this report. It is likely to matter most for the reserve arms, which leave the
  least VRAM free.
- **Decode** varied by up to 17% between starts of one arm (`ls_c` 32K: 78.6-91.6) and up to 45% between runs of one
  start. Arms whose decode range lies entirely outside their reference's: `tailskip_off` 32K (below) in the
  one-change table; `c_16k_1472` 32K (above) in the combinations; in the first 262K table several switch arms at 4K
  to 128K (above `base256`, 2 starts each); `c256_16k` 256K (below `ref256`). Everything else overlaps.
- **Answer quality:** the switches `STRATA_HIP_WMMA`, `STRATA_PF_FUSED=1`, `STRATA_SELECT_WMMA` and the tail-skip
  setting change output bits; this report checks only recall (18/18: 32K and 128K in the two precedent-method runs,
  256K in two 262K starts) and the pictures. The standard benchmark (12 sanity checks) of base, the `linux` set and
  `c256_16k_1472` is in a separate pull request.
- **Not measured:** GPU clocks, power and temperature during the runs; RAM speed; peak RAM and VRAM
  during the requests; contexts other than 131,072 and 262,144; concurrency; other quants; Linux. One machine and
  one day.
- **Not included:** the wrapper script and batch file that started the servers (local tools, not part of Strata).
  `data/starts.json` records, per start, the config, the switches, start and end time, background totals and the
  engine facts read from its log.

## Files

- [data/starts.json](data/starts.json): one record per start (arm, round, session, config, switches, background
  totals, slots, chunk, free VRAM, Windows commit, engine restarts, per-size runs, recall).
- [data/runs/](data/runs/): `benchmark.py`'s own output per start (`bench-output-<arm>-<round>.json`; `-full` for the
  precedent's method, `-L<n>` for 262K arms).
- [data/engine-logs/](data/engine-logs/): the engine log of each start, from its start line to its stop.
- [data/pictures.json](data/pictures.json): the picture questions, answers and timings.
- [configs/](configs/): the run configs used.

Local paths are shown as `<strata>` (the Strata folder) and `<data>` (the model data folder); environment variables
other than `STRATA_*` are left out.

Prepared with help from an AI assistant (Claude Opus 5.5); I ran every start, and the numbers were checked against the
files in this folder.
