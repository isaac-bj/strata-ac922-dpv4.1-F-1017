# IBM Power System AC922 (POWER9 + Tesla V100)

This is an experimental source-build target. It has been compiled on Debian
ppc64le with POWER9 CPUs, six Tesla V100 SXM2 16 GB cards, CUDA 12.4, GCC 12,
and CMake 3.25.

There is no prebuilt ppc64le package. Build on the AC922 and use the system's
CUDA toolkit. Volta requires the experimental SM70 path.

## DeepSeek-V4.1-Flash engine

```sh
git clone https://github.com/isaac-bj/strata-ac922-dpv4.1-F-1017.git Strata
cd Strata

cmake -S . -B build-ac922 \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_BUILD_DSV41=ON \
  -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_EXPERIMENTAL_SM60=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-ac922 --target strata-dsv41 -j 8
build-ac922/strata-dsv41 --version
```

`STRATA_NATIVE_EXPERTS=ON` needs a llama.cpp/ggml source tree. If it is not
already available under `third_party`, pass an isolated checkout with
`-DSTRATA_GGML_DIR=/absolute/path/to/llama.cpp`. During configuration Strata
normalizes two Power vector type spellings in that checkout for GCC 12, so do
not point this option at a checkout whose working tree must remain pristine.

The host compiler remains on C++20. CUDA translation units use C++17 on
ppc64le because CMake 3.25 does not advertise CUDA C++20 support for NVIDIA's
Power compiler driver.

## Model preparation

Point the installer at the source-built executable:

```sh
python3 -m venv .venv
STRATA_DSV41_EXE="$PWD/build-ac922/strata-dsv41" \
  .venv/bin/python setup.py \
  --family deepseek41 \
  --model IST-DAS-1.75bpw \
  --gpus 0,1,2,3,4,5 \
  --cuda 12 \
  --port 8081 \
  --no-browser
```

Use an explicit GPU list: `--gpus all` intentionally selects only GPUs in the
normal release support set, while V100 support is experimental. The 1.75 bpw
set downloads about 239 GB. Six 16 GB V100s hold roughly 90 GB of routed
experts; the remaining experts are computed by the POWER9 CPU from system RAM.

If another inference service already owns the V100s, use `--no-start` to finish
the download and pack first. Stop that service only when you are ready to give
the cards to Strata, then run the generated start script.

### Recommended six-V100 runtime layout

For the IST-DAS 1.75 bpw pack on six 16 GB V100s, keep a real CUDA workspace
instead of allowing the prompt chunk buffers to consume nearly all VRAM:

```text
--max-context 1048576
--vram-experts 1
--reserve-mib 2048
--prefill-chunk 0
```

This places roughly 76 GB of routed experts in VRAM and keeps roughly 46 GB in
POWER9 system memory, while leaving about 2 GB free on each GPU for inference
workspaces. On the tested AC922, the default 4096-token prompt chunk left only
tens of MiB free per card and failed at the first request. Windowed prefill
(`--prefill-chunk 0`) is slower for long prompts but stable on Volta. The 4096
context configuration measured about 19 prompt tokens/s and 21 decode tokens/s
on a cold short request, versus about 3.5 decode tokens/s with all routed
experts forced to the CPU. The native 1,048,576-token context uses the model's
compressed FP4 KV and sparse index caches (roughly 0.2--0.5 GB per card in the
tested layer split) and measured about 19 prompt tokens/s and 18 decode tokens/s
on the same cold short check.

## Notes

- CUDA and CPU device targets compile on ppc64le; x86-only expert kernels are
  excluded and portable/native fallbacks are linked in their place.
- The setup control path treats AVX/SSE as x86-only package-selection rules.
  POWER uses its native ISA and ggml's POWER9 backend.
- Keep the system CUDA driver/runtime on CUDA 12 for V100. CUDA 13 removes
  support for these older architectures.
- This port is source-only and experimental; published x86_64 release packages
  are not compatible with ppc64le.
