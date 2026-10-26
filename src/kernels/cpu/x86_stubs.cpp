// Portable definitions for optional x86-optimized expert kernels.
// Non-x86 builds use ggml-cpu's native row dots instead.
#include "strata/kernels/cpu/iq_avx1.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "strata/kernels/cpu/nvfp4_avx2.hpp"
#include "strata/kernels/cpu/nvfp4_avx512.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/tq_cpu.hpp"

#include <cstdlib>

namespace strata::kernels::cpu {

bool iq128_supported(int) noexcept { return false; }
bool iq128_down_supported(int) noexcept { return false; }
void iq128_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) { std::abort(); }
void iq128_down_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) { std::abort(); }
void q8k_quant_avx1(const float*, void*, int64_t) { std::abort(); }

bool iq256x_supported(int) noexcept { return false; }
int iq256_variant() noexcept { return 0; }
int iq256_variants() noexcept { return 0; }
void iq256_gu_rows_exact_one(int, const uint8_t*, size_t, size_t, int, const void*, float*, int, int) { std::abort(); }
void iq256_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) { std::abort(); }
void iq256_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) { std::abort(); }
void iq256x_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) { std::abort(); }
void iq256_gu_rows_v(int, int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) { std::abort(); }
void iq256_rows_v(int, int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) { std::abort(); }
void q8k_quant_avx2(const float*, void*, int64_t) { std::abort(); }
void iq4nl256_down_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) { std::abort(); }
void iq4nl256_down_rows_v(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) { std::abort(); }

bool iq512_gather_on() noexcept { return false; }
bool iq512x_supported(int) noexcept { return false; }
void iq512_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) { std::abort(); }
void iq512_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) { std::abort(); }
void iq512x_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) { std::abort(); }

void kq256_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) { std::abort(); }
void kq256_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) { std::abort(); }
void bf16_rows_dot(const uint16_t*, int, int, const float*, float*) { std::abort(); }
void bf16_rows_dot_multi(const uint16_t*, int, int, const float*, int, float*) { std::abort(); }

bool nvfp4_256_fits(int) { return false; }
void nvfp4_256_gu_rows(const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int, float, float) { std::abort(); }
void nvfp4_256_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int, float) { std::abort(); }
bool nvfp4_512_fits(int) { return false; }
void nvfp4_512_gu_rows(const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int, float, float) { std::abort(); }
void nvfp4_512_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int, float) { std::abort(); }

void q2_0_gguf_rows_multi(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) { std::abort(); }
void q2_0_gguf_rows_multi_avx2(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) { std::abort(); }
void q2_0_gguf_rows_multi_avx2_legacy(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) { std::abort(); }
void q2_0_gguf_rows_multi_avx1(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) { std::abort(); }
void q2_0_gguf_rows_multi_avx1_legacy(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) { std::abort(); }
void act_quant_q8_1(const float*, int, ActQ&) { std::abort(); }
void act_quant_q8_1_avx2(const float*, int, ActQ&) { std::abort(); }
void act_quant_q8_1_avx1(const float*, int, ActQ&) { std::abort(); }

namespace tq_detail {
void tq_rows_avx2(int, const int16_t*, const uint8_t*, size_t, int, const TqAct* const*, int, float* const*, int, int) { std::abort(); }
void tq_rows_avx512(int, const int16_t*, const uint8_t*, size_t, int, const TqAct* const*, int, float* const*, int, int) { std::abort(); }
}  // namespace tq_detail

}  // namespace strata::kernels::cpu
