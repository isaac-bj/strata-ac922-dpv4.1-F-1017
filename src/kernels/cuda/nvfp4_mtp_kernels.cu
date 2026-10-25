// src/kernels/cuda/nvfp4_mtp_kernels.cu - RDNA4 (gfx1201) HIP kernels for native Multi-Token Prediction (MTP),
// target logit confidence/margin extraction, NSA/CSA sink+window pinning, and capped hybrid small-batch MoE routing.
#include "strata/kernels/nvfp4_mtp_kernels.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace strata::kernels {
namespace {

__device__ __forceinline__ float bf16_to_f32(uint16_t b) {
    return __uint_as_float(static_cast<uint32_t>(b) << 16);
}

__device__ __forceinline__ uint16_t f32_to_bf16(float v) {
    uint32_t bits = __float_as_uint(v);
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7FFFu + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

__device__ __forceinline__ float fp8_e4m3_to_f32(uint32_t b) {
    uint32_t sign = (b & 0x80u) << 24;
    uint32_t exp = (b >> 3) & 0xFu;
    uint32_t man = b & 0x7u;
    float mag = (exp == 0u)
                    ? (static_cast<float>(man) * 0.001953125f)
                    : __uint_as_float(((exp + 120u) << 23) | (man << 20));
    return __uint_as_float(__float_as_uint(mag) | sign);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, m, 32);
    }
    return v;
}

__global__ void nvfp4_mtp_ring_write_k(const uint16_t* __restrict__ src_x,
                                       uint16_t* __restrict__ dst_ring,
                                       int32_t pos0, int32_t ring_mask) {
    const int32_t t = static_cast<int32_t>(blockIdx.y);
    const int32_t slot = (pos0 + t) & ring_mask;
    const uint4* src4 = reinterpret_cast<const uint4*>(src_x + static_cast<size_t>(t) * 10240);
    uint4* dst4 = reinterpret_cast<uint4*>(dst_ring + static_cast<size_t>(slot) * 10240);
    const int32_t idx = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < 1280) {
        dst4[idx] = src4[idx];
    }
}

__global__ void nvfp4_mtp_emb_rms_norm_k(const int32_t* __restrict__ tok_ids,
                                         const uint16_t* __restrict__ embed_tokens,
                                         const uint16_t* __restrict__ norm_w,
                                         uint16_t* __restrict__ out_bf16,
                                         float eps) {
    const int32_t t = static_cast<int32_t>(blockIdx.x);
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    int32_t tok = tok_ids[t];
    if (tok < 0 || tok >= 248320) tok = 0;

    const uint16_t* row = embed_tokens + static_cast<size_t>(tok) * 2560;
    uint16_t* out_row = out_bf16 + static_cast<size_t>(t) * 2560;

    float vals[10];
    float sq_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 10; ++i) {
        int c = tid + i * 256;
        float v = bf16_to_f32(row[c]);
        vals[i] = v;
        sq_sum += v * v;
    }

    sq_sum = warp_sum(sq_sum);
    __shared__ float s_warp[8];
    const int lane = tid & 31;
    const int wid = tid >> 5;
    if (lane == 0) s_warp[wid] = sq_sum;
    __syncthreads();

    if (wid == 0) {
        float acc = (lane < 8) ? s_warp[lane] : 0.0f;
        acc = warp_sum(acc);
        if (lane == 0) {
            s_warp[0] = rsqrtf(acc * (1.0f / 2560.0f) + eps);
        }
    }
    __syncthreads();
    const float rstd = s_warp[0];

#pragma unroll
    for (int i = 0; i < 10; ++i) {
        int c = tid + i * 256;
        float w = 1.0f + bf16_to_f32(norm_w[c]);
        out_row[c] = f32_to_bf16(vals[i] * rstd * w);
    }
}

__global__ void nvfp4_mtp_hidden_rms_norm_k(const uint16_t* __restrict__ r_base,
                                            int32_t ring_pos0, int32_t ring_mask,
                                            const uint16_t* __restrict__ norm_w,
                                            uint16_t* __restrict__ out_bf16,
                                            float eps, int32_t per_stream) {
    const int32_t t = static_cast<int32_t>(blockIdx.x);
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    const int32_t slot = (ring_mask > 0) ? ((ring_pos0 + t) & ring_mask) : t;
    const uint16_t* in_row = r_base + static_cast<size_t>(slot) * 10240;
    uint16_t* out_row = out_bf16 + static_cast<size_t>(t) * 10240;

    __shared__ float s_warp[8];
    const int lane = tid & 31;
    const int wid = tid >> 5;

    if (!per_stream) {
        float sq_sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 40; ++i) {
            int c = tid + i * 256;
            float v = bf16_to_f32(in_row[c]);
            sq_sum += v * v;
        }
        sq_sum = warp_sum(sq_sum);
        if (lane == 0) s_warp[wid] = sq_sum;
        __syncthreads();
        if (wid == 0) {
            float acc = (lane < 8) ? s_warp[lane] : 0.0f;
            acc = warp_sum(acc);
            if (lane == 0) s_warp[0] = rsqrtf(acc * (1.0f / 10240.0f) + eps);
        }
        __syncthreads();
        const float rstd = s_warp[0];
#pragma unroll
        for (int i = 0; i < 40; ++i) {
            int c = tid + i * 256;
            float v = bf16_to_f32(in_row[c]);
            float w = 1.0f + bf16_to_f32(norm_w[c]);
            out_row[c] = f32_to_bf16(v * rstd * w);
        }
    } else {
        for (int s = 0; s < 4; ++s) {
            const int base_c = s * 2560;
            float sq_sum = 0.0f;
#pragma unroll
            for (int i = 0; i < 10; ++i) {
                int c = base_c + tid + i * 256;
                float v = bf16_to_f32(in_row[c]);
                sq_sum += v * v;
            }
            sq_sum = warp_sum(sq_sum);
            if (lane == 0) s_warp[wid] = sq_sum;
            __syncthreads();
            if (wid == 0) {
                float acc = (lane < 8) ? s_warp[lane] : 0.0f;
                acc = warp_sum(acc);
                if (lane == 0) s_warp[0] = rsqrtf(acc * (1.0f / 2560.0f) + eps);
            }
            __syncthreads();
            const float rstd = s_warp[0];
#pragma unroll
            for (int i = 0; i < 10; ++i) {
                int c = base_c + tid + i * 256;
                float v = bf16_to_f32(in_row[c]);
                float w = 1.0f + bf16_to_f32(norm_w[c]);
                out_row[c] = f32_to_bf16(v * rstd * w);
            }
            __syncthreads();
        }
    }
}

__global__ void nvfp4_mtp_stream_broadcast_add_k(const uint16_t* __restrict__ h2,
                                                 const uint16_t* __restrict__ e2,
                                                 uint16_t* __restrict__ r_out) {
    const int32_t t = static_cast<int32_t>(blockIdx.y);
    const int32_t idx = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < 10240) {
        const int32_t c = idx % 2560;
        const size_t off_h = static_cast<size_t>(t) * 10240 + static_cast<size_t>(idx);
        const size_t off_e = static_cast<size_t>(t) * 2560 + static_cast<size_t>(c);
        float vh = bf16_to_f32(h2[off_h]);
        float ve = bf16_to_f32(e2[off_e]);
        r_out[off_h] = f32_to_bf16(vh + ve);
    }
}

__global__ void nvfp4_mtp_kv_idx_proj_k(const uint16_t* __restrict__ x,
                                        const uint16_t* __restrict__ w_kv_idx,
                                        uint16_t* __restrict__ k_out,
                                        uint16_t* __restrict__ v_out,
                                        uint16_t* __restrict__ idx_qk_out) {
    __shared__ float s_x[2560];
    const int32_t t = static_cast<int32_t>(blockIdx.y);
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    const uint16_t* x_row = x + static_cast<size_t>(t) * 2560;

#pragma unroll
    for (int i = 0; i < 10; ++i) {
        int c = tid + i * 256;
        s_x[c] = bf16_to_f32(x_row[c]);
    }
    __syncthreads();

    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int32_t r = static_cast<int32_t>(blockIdx.x) * 8 + wid;
    if (r >= 1664) return;

    const uint4* w4 = reinterpret_cast<const uint4*>(w_kv_idx + static_cast<size_t>(r) * 2560);
    float acc = 0.0f;
#pragma unroll
    for (int step = 0; step < 10; ++step) {
        int c4 = step * 32 + lane;
        uint4 pack = w4[c4];
        const uint16_t* u16 = reinterpret_cast<const uint16_t*>(&pack);
        int base_c = c4 * 8;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc += bf16_to_f32(u16[j]) * s_x[base_c + j];
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        uint16_t out_val = f32_to_bf16(acc);
        if (r < 512) {
            k_out[static_cast<size_t>(t) * 512 + r] = out_val;
        } else if (r < 1024) {
            v_out[static_cast<size_t>(t) * 512 + (r - 512)] = out_val;
        } else {
            idx_qk_out[static_cast<size_t>(t) * 640 + (r - 1024)] = out_val;
        }
    }
}

__global__ void nvfp4_mtp_moe_decode_gate_up_k(const uint16_t* __restrict__ mixed_bf16,
                                               const int32_t* __restrict__ cur_ids,
                                               const int64_t* __restrict__ ptrs,
                                               uint16_t* __restrict__ moe_inter_bf16) {
    const int32_t k = static_cast<int32_t>(blockIdx.y);
    const int32_t raw_id = cur_ids[k];
    if (raw_id < 0 || (raw_id & 0x10000) != 0) return;
    const int32_t e = raw_id & 0xFFFF;
    if (e < 0 || e >= 512) return;

    const uint8_t* gw = reinterpret_cast<const uint8_t*>(ptrs[e * 6 + 0]);
    if (!gw) return;
    const uint16_t* gs = reinterpret_cast<const uint16_t*>(ptrs[e * 6 + 1]);
    const uint8_t* uw = reinterpret_cast<const uint8_t*>(ptrs[e * 6 + 2]);
    const uint16_t* us = reinterpret_cast<const uint16_t*>(ptrs[e * 6 + 3]);

    __shared__ float s_x[2560];
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
#pragma unroll
    for (int i = 0; i < 10; ++i) {
        int c = tid + i * 256;
        s_x[c] = bf16_to_f32(mixed_bf16[c]);
    }
    __syncthreads();

    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int32_t r = static_cast<int32_t>(blockIdx.x) * 8 + wid;
    if (r >= 640) return;

    const int32_t rb = r >> 7;  // r / 128 in [0..4]
    const uint32_t* gw4 = reinterpret_cast<const uint32_t*>(gw + static_cast<size_t>(r) * 2560);
    const uint32_t* uw4 = reinterpret_cast<const uint32_t*>(uw + static_cast<size_t>(r) * 2560);

    float g_acc = 0.0f;
    float u_acc = 0.0f;
#pragma unroll
    for (int b = 0; b < 20; ++b) {
        uint32_t gp = gw4[b * 32 + lane];
        uint32_t up = uw4[b * 32 + lane];
        int base_c = b * 128 + lane * 4;
        float x0 = s_x[base_c + 0];
        float x1 = s_x[base_c + 1];
        float x2 = s_x[base_c + 2];
        float x3 = s_x[base_c + 3];

        float g_sub = fp8_e4m3_to_f32(gp & 0xFFu) * x0 +
                      fp8_e4m3_to_f32((gp >> 8) & 0xFFu) * x1 +
                      fp8_e4m3_to_f32((gp >> 16) & 0xFFu) * x2 +
                      fp8_e4m3_to_f32((gp >> 24) & 0xFFu) * x3;
        float u_sub = fp8_e4m3_to_f32(up & 0xFFu) * x0 +
                      fp8_e4m3_to_f32((up >> 8) & 0xFFu) * x1 +
                      fp8_e4m3_to_f32((up >> 16) & 0xFFu) * x2 +
                      fp8_e4m3_to_f32((up >> 24) & 0xFFu) * x3;

        float sc_g = bf16_to_f32(gs[rb * 20 + b]);
        float sc_u = bf16_to_f32(us[rb * 20 + b]);
        g_acc += g_sub * sc_g;
        u_acc += u_sub * sc_u;
    }

    g_acc = warp_sum(g_acc);
    u_acc = warp_sum(u_acc);
    if (lane == 0) {
        float gf = bf16_to_f32(f32_to_bf16(g_acc));
        float uf = bf16_to_f32(f32_to_bf16(u_acc));
        float silu_g = bf16_to_f32(f32_to_bf16(gf / (1.0f + __expf(-gf))));
        moe_inter_bf16[static_cast<size_t>(k) * 640 + r] = f32_to_bf16(silu_g * uf);
    }
}

__global__ void nvfp4_mtp_moe_decode_down_k(const uint16_t* __restrict__ moe_inter_bf16,
                                            const int32_t* __restrict__ cur_ids,
                                            const int64_t* __restrict__ ptrs,
                                            uint16_t* __restrict__ moe_y_bf16) {
    const int32_t k = static_cast<int32_t>(blockIdx.y);
    const int32_t raw_id = cur_ids[k];
    if (raw_id < 0 || (raw_id & 0x10000) != 0) return;
    const int32_t e = raw_id & 0xFFFF;
    if (e < 0 || e >= 512) return;

    const uint8_t* dw = reinterpret_cast<const uint8_t*>(ptrs[e * 6 + 4]);
    if (!dw) return;
    const uint16_t* ds = reinterpret_cast<const uint16_t*>(ptrs[e * 6 + 5]);

    __shared__ float s_h[640];
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    const uint16_t* h_row = moe_inter_bf16 + static_cast<size_t>(k) * 640;
    for (int c = tid; c < 640; c += 256) {
        s_h[c] = bf16_to_f32(h_row[c]);
    }
    __syncthreads();

    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int32_t r = static_cast<int32_t>(blockIdx.x) * 8 + wid;
    if (r >= 2560) return;

    const int32_t rb = r >> 7;  // r / 128 in [0..19]
    const uint32_t* dw4 = reinterpret_cast<const uint32_t*>(dw + static_cast<size_t>(r) * 640);

    float d_acc = 0.0f;
#pragma unroll
    for (int b = 0; b < 5; ++b) {
        uint32_t dp = dw4[b * 32 + lane];
        int base_c = b * 128 + lane * 4;
        float d_sub = fp8_e4m3_to_f32(dp & 0xFFu) * s_h[base_c + 0] +
                      fp8_e4m3_to_f32((dp >> 8) & 0xFFu) * s_h[base_c + 1] +
                      fp8_e4m3_to_f32((dp >> 16) & 0xFFu) * s_h[base_c + 2] +
                      fp8_e4m3_to_f32((dp >> 24) & 0xFFu) * s_h[base_c + 3];
        float sc_d = bf16_to_f32(ds[rb * 5 + b]);
        d_acc += d_sub * sc_d;
    }

    d_acc = warp_sum(d_acc);
    if (lane == 0) {
        moe_y_bf16[static_cast<size_t>(k) * 2560 + r] = f32_to_bf16(d_acc);
    }
}

__device__ __forceinline__ void merge_online_top2_softmax(
    float& m1, int32_t& id1, float& m2, float& s,
    float om1, int32_t oid1, float om2, float os) {
    if (om1 > m1 || (om1 == m1 && oid1 < id1)) {
        float new_m2 = fmaxf(m1, om2);
        float new_s = os + s * __expf(m1 - om1);
        m1 = om1;
        id1 = oid1;
        m2 = new_m2;
        s = new_s;
    } else {
        m2 = fmaxf(m2, om1);
        s = s + os * __expf(om1 - m1);
    }
}

__global__ void nvfp4_argmax_prob_margin_248320_k(const uint16_t* __restrict__ logits_bf16,
                                                  int32_t* __restrict__ out_tok,
                                                  Nvfp4ArgmaxSummary* __restrict__ out_summary) {
    const int32_t row = static_cast<int32_t>(blockIdx.x);
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    const uint4* row4 = reinterpret_cast<const uint4*>(logits_bf16 + static_cast<size_t>(row) * 248320);

    float m1 = -1e30f;
    int32_t id1 = 0;
    float m2 = -1e30f;
    float s = 0.0f;

    // 248320 / 8 = 31040 uint4 loads across 1024 threads (~30.3 loads per thread)
    for (int32_t i4 = tid; i4 < 31040; i4 += 1024) {
        uint4 pack = row4[i4];
        const uint16_t* u16 = reinterpret_cast<const uint16_t*>(&pack);
        int32_t base_idx = i4 * 8;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float v = bf16_to_f32(u16[j]);
            int32_t idx = base_idx + j;
            if (v > m1) {
                m2 = m1;
                s = s * __expf(m1 - v) + 1.0f;
                m1 = v;
                id1 = idx;
            } else {
                if (v > m2) m2 = v;
                s += __expf(v - m1);
            }
        }
    }

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        float om1 = __shfl_xor_sync(0xffffffffu, m1, mask, 32);
        int32_t oid1 = __shfl_xor_sync(0xffffffffu, id1, mask, 32);
        float om2 = __shfl_xor_sync(0xffffffffu, m2, mask, 32);
        float os = __shfl_xor_sync(0xffffffffu, s, mask, 32);
        merge_online_top2_softmax(m1, id1, m2, s, om1, oid1, om2, os);
    }

    __shared__ float sm1[32];
    __shared__ int32_t sid1[32];
    __shared__ float sm2[32];
    __shared__ float ss[32];

    const int lane = tid & 31;
    const int wid = tid >> 5;
    if (lane == 0) {
        sm1[wid] = m1;
        sid1[wid] = id1;
        sm2[wid] = m2;
        ss[wid] = s;
    }
    __syncthreads();

    if (wid == 0) {
        m1 = sm1[lane];
        id1 = sid1[lane];
        m2 = sm2[lane];
        s = ss[lane];
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            float om1 = __shfl_xor_sync(0xffffffffu, m1, mask, 32);
            int32_t oid1 = __shfl_xor_sync(0xffffffffu, id1, mask, 32);
            float om2 = __shfl_xor_sync(0xffffffffu, m2, mask, 32);
            float os = __shfl_xor_sync(0xffffffffu, s, mask, 32);
            merge_online_top2_softmax(m1, id1, m2, s, om1, oid1, om2, os);
        }
        if (lane == 0) {
            if (out_tok) out_tok[row] = id1;
            if (out_summary) {
                out_summary[row].tok_id = id1;
                out_summary[row].prob = 1.0f / fmaxf(s, 1.0f);
                out_summary[row].margin = fmaxf(0.0f, m1 - m2);
            }
        }
    }
}

__global__ void nvfp4_qsa_pin_sink_and_window_k(float* __restrict__ scores,
                                                int32_t pos0, int32_t score_stride,
                                                int32_t n_sink, int32_t n_local) {
    const int32_t t = static_cast<int32_t>(blockIdx.y);
    const int32_t pos = pos0 + t;
    const int32_t nb = (pos + 1) >> 2;
    float* row_scores = scores + static_cast<size_t>(t) * static_cast<size_t>(score_stride);
    const int32_t tid = static_cast<int32_t>(threadIdx.x);

    // Pin initial attention sink blocks [0, n_sink)
    if (tid < n_sink && tid < nb) {
        row_scores[tid] = 1e30f;
    }
    // Pin local sliding window blocks [nb - n_local, nb)
    int32_t local_start = nb - n_local;
    if (local_start < n_sink) local_start = n_sink;
    int32_t b = local_start + tid;
    if (b >= local_start && b < nb) {
        row_scores[b] = 1e30f;
    }
}

__global__ void nvfp4_moe_cap_gpu_cold_k(int32_t* __restrict__ cur_ids,
                                         int32_t* __restrict__ gpu_align_ids,
                                         const int64_t* __restrict__ ptrs_decode,
                                         const int64_t* __restrict__ ptrs_vram,
                                         int32_t pairs, int32_t max_gpu_cold) {
    if (threadIdx.x != 0) return;
    int32_t cold_e[16];
    int32_t n_cold = 0;
    for (int32_t p = 0; p < pairs; ++p) {
        int32_t raw = cur_ids[p];
        int32_t e = (raw >= 0) ? (raw & 0xFFFF) : -1;
        if (e < 0 || e >= 512) {
            gpu_align_ids[p] = -1;
            continue;
        }
        if (ptrs_vram[e * 6] != 0) {
            // Resident in VRAM: run on GPU with 0 PCIe cost
            cur_ids[p] = e;
            gpu_align_ids[p] = e;
            continue;
        }
        if (ptrs_decode[e * 6] == 0) {
            // CPU-only cold expert: mark for CPU pool and exclude from GPU moe_align
            cur_ids[p] = e | 0x10000;
            gpu_align_ids[p] = -1;
            continue;
        }
        // GPU-cold PCIe expert: check if already admitted in this layer's batch
        bool seen = false;
        for (int32_t k = 0; k < n_cold; ++k) {
            if (cold_e[k] == e) {
                seen = true;
                break;
            }
        }
        if (seen || n_cold < max_gpu_cold) {
            if (!seen && n_cold < 16) cold_e[n_cold++] = e;
            cur_ids[p] = e;
            gpu_align_ids[p] = e;
        } else {
            // Cap exceeded: overflow to 16-core AVX-512 CPU pool!
            cur_ids[p] = e | 0x10000;
            gpu_align_ids[p] = -1;
        }
    }
}

__global__ __launch_bounds__(256) void nvfp4_mtp_dvocab_gemv_k(
    const uint16_t* __restrict__ x_bf16,
    const uint16_t* __restrict__ lm_head_bf16,
    const int32_t* __restrict__ dvocab,
    uint16_t* __restrict__ logits_out_bf16,
    int32_t n_dvocab) {
    __shared__ uint4 s_x4[320]; // 2560 BF16 = 320 uint4 (5120 B)
    const int tid = static_cast<int>(threadIdx.x);
    const uint4* x4 = reinterpret_cast<const uint4*>(x_bf16);
    for (int i = tid; i < 320; i += 256) {
        s_x4[i] = x4[i];
    }
    __syncthreads();

    const int wid = tid >> 5;   // 0..7 (8 rows per 256-thread block)
    const int lane = tid & 31;  // 0..31
    const int32_t row_idx = static_cast<int32_t>(blockIdx.x) * 8 + wid;
    if (row_idx >= n_dvocab) return;

    const int32_t tok_id = dvocab[row_idx];
    const uint4* w_row4 = reinterpret_cast<const uint4*>(
        lm_head_bf16 + static_cast<size_t>(tok_id) * 2560ULL);

    float acc = 0.0f;
#pragma unroll
    for (int step = 0; step < 10; ++step) {
        const int idx4 = step * 32 + lane;
        uint4 wv = w_row4[idx4];
        uint4 xv = s_x4[idx4];
        const uint16_t* wu = reinterpret_cast<const uint16_t*>(&wv);
        const uint16_t* xu = reinterpret_cast<const uint16_t*>(&xv);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            acc += bf16_to_f32(wu[k]) * bf16_to_f32(xu[k]);
        }
    }

    acc = warp_sum(acc);
    if (lane == 0) {
        logits_out_bf16[row_idx] = f32_to_bf16(acc);
    }
}

__global__ __launch_bounds__(1024) void nvfp4_argmax_prob_margin_dvocab_k(
    const uint16_t* __restrict__ logits_bf16,
    const int32_t* __restrict__ dvocab,
    int32_t n_dvocab,
    int32_t* __restrict__ out_tok,
    Nvfp4ArgmaxSummary* __restrict__ out_summary) {
    const int32_t tid = static_cast<int32_t>(threadIdx.x);

    float m1 = -1e30f;
    int32_t id1 = 0;
    float m2 = -1e30f;
    float s = 0.0f;

    for (int32_t idx = tid; idx < n_dvocab; idx += 1024) {
        float v = bf16_to_f32(logits_bf16[idx]);
        if (v > m1) {
            m2 = m1;
            s = s * __expf(m1 - v) + 1.0f;
            m1 = v;
            id1 = idx;
        } else {
            if (v > m2) m2 = v;
            s += __expf(v - m1);
        }
    }

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        float om1 = __shfl_xor_sync(0xffffffffu, m1, mask, 32);
        int32_t oid1 = __shfl_xor_sync(0xffffffffu, id1, mask, 32);
        float om2 = __shfl_xor_sync(0xffffffffu, m2, mask, 32);
        float os = __shfl_xor_sync(0xffffffffu, s, mask, 32);
        merge_online_top2_softmax(m1, id1, m2, s, om1, oid1, om2, os);
    }

    __shared__ float sm1[32];
    __shared__ int32_t sid1[32];
    __shared__ float sm2[32];
    __shared__ float ss[32];

    const int lane = tid & 31;
    const int wid = tid >> 5;
    if (lane == 0) {
        sm1[wid] = m1;
        sid1[wid] = id1;
        sm2[wid] = m2;
        ss[wid] = s;
    }
    __syncthreads();

    if (wid == 0) {
        m1 = sm1[lane];
        id1 = sid1[lane];
        m2 = sm2[lane];
        s = ss[lane];
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            float om1 = __shfl_xor_sync(0xffffffffu, m1, mask, 32);
            int32_t oid1 = __shfl_xor_sync(0xffffffffu, id1, mask, 32);
            float om2 = __shfl_xor_sync(0xffffffffu, m2, mask, 32);
            float os = __shfl_xor_sync(0xffffffffu, s, mask, 32);
            merge_online_top2_softmax(m1, id1, m2, s, om1, oid1, om2, os);
        }
        if (lane == 0) {
            int32_t true_tok = (id1 >= 0 && id1 < n_dvocab) ? dvocab[id1] : id1;
            if (out_tok) out_tok[0] = true_tok;
            if (out_summary) {
                out_summary[0].tok_id = true_tok;
                out_summary[0].prob = 1.0f / fmaxf(s, 1.0f);
                out_summary[0].margin = fmaxf(0.0f, m1 - m2);
            }
        }
    }
}

__global__ void nvfp4_mtp_topk_truncate_k(int32_t* __restrict__ cur_ids,
                                          float* __restrict__ cur_wts,
                                          int32_t keep_k) {
    const int lane = static_cast<int>(threadIdx.x);
    if (lane >= 32) return;
    int32_t my_id = (lane < 10) ? cur_ids[lane] : -1;
    float my_w = (lane < 10 && my_id >= 0) ? cur_wts[lane] : -1.0f;

    // Compute rank of each of the 10 experts by weight descending
    int rank = 0;
#pragma unroll
    for (int j = 0; j < 10; ++j) {
        float wj = __shfl_sync(0xffffffffu, my_w, j, 32);
        if (wj > my_w || (wj == my_w && j < lane)) {
            rank++;
        }
    }

    float orig_w = (lane < 10 && my_w > 0.0f) ? my_w : 0.0f;
    float kept_w = (lane < 10 && rank < keep_k && my_w > 0.0f) ? my_w : 0.0f;
    float sum_orig = warp_sum(orig_w);
    float sum_kept = warp_sum(kept_w);

    if (lane < 10) {
        if (rank < keep_k && my_id >= 0) {
            float scale = (sum_kept > 1e-8f) ? (sum_orig / sum_kept) : 1.0f;
            cur_wts[lane] = my_w * scale;
        } else {
            cur_ids[lane] = -1;
            cur_wts[lane] = 0.0f;
        }
    }
}

__device__ __forceinline__ float bf_round(float v) {
    return bf16_to_f32(f32_to_bf16(v));
}

__device__ __forceinline__ void idx_norm_rope_128(
    const float vals[4], int lane, const float w4[4],
    const uint16_t* __restrict__ cos_ptr, const uint16_t* __restrict__ sin_ptr,
    int32_t pos, float eps, float out[4]) {
    float ss = vals[0] * vals[0] + vals[1] * vals[1] + vals[2] * vals[2] + vals[3] * vals[3];
    ss = warp_sum(ss);
    float rstd = rsqrtf(ss * (1.0f / 128.0f) + eps);
    float h[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        h[e] = bf_round(vals[e] * rstd * (1.0f + w4[e]));
    }
    float partner[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        partner[e] = __shfl_xor_sync(0xffffffffu, h[e], 8, 32);
    }
    const bool in_rot = (lane < 16);
    const bool first = (lane < 8);
    const int32_t d0 = in_rot ? (lane * 4) : 0;
    const size_t roff = static_cast<size_t>(pos) * 64ULL + static_cast<size_t>(d0);
    const uint2 cs2 = *reinterpret_cast<const uint2*>(cos_ptr + roff);
    const uint2 sn2 = *reinterpret_cast<const uint2*>(sin_ptr + roff);
    const uint16_t* csu = reinterpret_cast<const uint16_t*>(&cs2);
    const uint16_t* snu = reinterpret_cast<const uint16_t*>(&sn2);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        float cs = bf16_to_f32(csu[e]);
        float sn = bf16_to_f32(snu[e]);
        float rot = first ? -partner[e] : partner[e];
        float r = bf_round(bf_round(h[e] * cs) + bf_round(rot * sn));
        out[e] = in_rot ? r : h[e];
    }
}

__global__ __launch_bounds__(128) void nvfp4_qsa_block_keys_from_batch_k(
    uint16_t* __restrict__ rk_ptr,
    const uint16_t* __restrict__ idx_qk_b,
    const uint16_t* __restrict__ kw_ptr,
    const uint16_t* __restrict__ cos_ptr,
    const uint16_t* __restrict__ sin_ptr,
    uint16_t* __restrict__ bk_ptr,
    int64_t bk_host_delta_elems,
    int32_t pos0, int32_t M,
    int32_t b_start, int32_t b_cnt,
    int32_t vram_blocks, float eps) {
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    const int32_t lane = tid & 31;
    const int32_t wave = tid >> 5;
    const int32_t i = static_cast<int32_t>(blockIdx.x) * 4 + wave;

    if (i < b_cnt) {
        const int32_t blk = b_start + i;
        const int32_t pos = blk * 4;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int32_t tok = pos + r;
            const uint16_t* src = (tok < pos0)
                                      ? (rk_ptr + static_cast<size_t>(tok & 1023) * 128ULL + lane * 4)
                                      : (idx_qk_b + static_cast<size_t>(tok - pos0) * 640ULL + 512ULL + lane * 4);
            uint2 v2 = *reinterpret_cast<const uint2*>(src);
            const uint16_t* vu = reinterpret_cast<const uint16_t*>(&v2);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                acc[e] += bf16_to_f32(vu[e]);
            }
        }
        float pooled[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            pooled[e] = bf_round(acc[e] * 0.25f);
        }
        uint2 w2 = *reinterpret_cast<const uint2*>(kw_ptr + lane * 4);
        const uint16_t* wu = reinterpret_cast<const uint16_t*>(&w2);
        float w4[4] = {bf16_to_f32(wu[0]), bf16_to_f32(wu[1]), bf16_to_f32(wu[2]), bf16_to_f32(wu[3])};
        float out[4];
        idx_norm_rope_128(pooled, lane, w4, cos_ptr, sin_ptr, pos, eps, out);

        const bool in_vram = (blk < vram_blocks);
        const int32_t slot = in_vram ? blk : (blk - vram_blocks);
        const int64_t bk_off = static_cast<int64_t>(slot) * 128LL + lane * 4 + (in_vram ? 0LL : bk_host_delta_elems);
        uint16_t out_u16[4] = {f32_to_bf16(out[0]), f32_to_bf16(out[1]), f32_to_bf16(out[2]), f32_to_bf16(out[3])};
        *reinterpret_cast<uint2*>(bk_ptr + bk_off) = *reinterpret_cast<const uint2*>(out_u16);
    }

    // Deterministically refresh the last min(M, 4) slots of RAW_K_CACHE from idx_qk_b
    // so any subsequent prefill chunk or decode step sees exact unpooled keys even when M > 1024.
    if (blockIdx.x == 0 && M > 0) {
        const int32_t t_tail_start = (M > 4) ? (M - 4) : 0;
        for (int32_t t = t_tail_start; t < M; ++t) {
            const int32_t rk_slot = (pos0 + t) & 1023;
            rk_ptr[rk_slot * 128 + tid] = idx_qk_b[static_cast<size_t>(t) * 640ULL + 512ULL + tid];
        }
    }
}

__global__ __launch_bounds__(256) void nvfp4_qsa_score_blocks_batched_k(
    const uint16_t* __restrict__ qk_ptr,
    const uint16_t* __restrict__ qw_ptr,
    const uint16_t* __restrict__ cos_ptr,
    const uint16_t* __restrict__ sin_ptr,
    const uint16_t* __restrict__ bk_ptr,
    float* __restrict__ scores_ptr,
    int64_t bk_host_delta_elems,
    int32_t pos_cur, int32_t m_cur,
    int32_t vram_blocks, int32_t score_stride,
    int32_t n_sink, int32_t n_local, float eps) {
    __shared__ float q_s[8][512];

    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    const int32_t wave = tid >> 5;   // 0..7 -> 1 wave per query token in the 8-token tile
    const int32_t lane = tid & 31;
    const int32_t t_base = static_cast<int32_t>(blockIdx.y) * 8;
    const int32_t k_cnt = (t_base + 8 <= m_cur) ? 8 : (m_cur - t_base);

    if (wave < k_cnt) {
        const int32_t t_rel = t_base + wave;
        const int32_t pos_t = pos_cur + t_rel;
        const uint16_t* qk_row = qk_ptr + static_cast<size_t>(t_rel) * 640ULL;
        uint2 w2 = *reinterpret_cast<const uint2*>(qw_ptr + lane * 4);
        const uint16_t* wu = reinterpret_cast<const uint16_t*>(&w2);
        float w4[4] = {bf16_to_f32(wu[0]), bf16_to_f32(wu[1]), bf16_to_f32(wu[2]), bf16_to_f32(wu[3])};
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            const int32_t hq = h * 128 + lane * 4;
            uint2 q2 = *reinterpret_cast<const uint2*>(qk_row + hq);
            const uint16_t* qu = reinterpret_cast<const uint16_t*>(&q2);
            float qv[4] = {bf16_to_f32(qu[0]), bf16_to_f32(qu[1]), bf16_to_f32(qu[2]), bf16_to_f32(qu[3])};
            float qn[4];
            idx_norm_rope_128(qv, lane, w4, cos_ptr, sin_ptr, pos_t, eps, qn);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                q_s[wave][hq + e] = qn[e];
            }
        }
    }
    __syncthreads();

    const int32_t b = static_cast<int32_t>(blockIdx.x) * 256 + tid;
    const int32_t max_pos_tile = pos_cur + t_base + k_cnt - 1;
    const int32_t max_nb_tile = (max_pos_tile + 1) >> 2;
    if (b >= max_nb_tile) return;

    const bool in_vram = (b < vram_blocks);
    const int32_t slot = in_vram ? b : (b - vram_blocks);
    const int64_t kbase = static_cast<int64_t>(slot) * 128LL + (in_vram ? 0LL : bk_host_delta_elems);
    const uint4* bk4 = reinterpret_cast<const uint4*>(bk_ptr + kbase);

    float dots[8][4];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            dots[k][h] = 0.0f;
        }
    }

#pragma unroll
    for (int c = 0; c < 16; ++c) {
        uint4 pack = bk4[c];
        const uint16_t* ku = reinterpret_cast<const uint16_t*>(&pack);
        const int base_d = c * 8;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            float kf = bf16_to_f32(ku[e]);
            const int d = base_d + e;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                if (k < k_cnt) {
                    dots[k][0] += q_s[k][d] * kf;
                    dots[k][1] += q_s[k][128 + d] * kf;
                    dots[k][2] += q_s[k][256 + d] * kf;
                    dots[k][3] += q_s[k][384 + d] * kf;
                }
            }
        }
    }

    static constexpr float kInvSqrt128 = 0.08838834764831845f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (k < k_cnt) {
            const int32_t t_rel = t_base + k;
            const int32_t nb_k = (pos_cur + t_rel + 1) >> 2;
            if (b < nb_k) {
                float sc;
                if (b < n_sink || b >= nb_k - n_local) {
                    sc = 1e30f;
                } else {
                    sc = (fmaxf(dots[k][0], 0.0f) + fmaxf(dots[k][1], 0.0f) +
                          fmaxf(dots[k][2], 0.0f) + fmaxf(dots[k][3], 0.0f)) *
                         kInvSqrt128;
                }
                scores_ptr[static_cast<size_t>(t_rel) * static_cast<size_t>(score_stride) + b] = sc;
            }
        }
    }
}

}  // namespace

void launch_nvfp4_mtp_ring_write(const void* src_x_bf16, void* dst_ring_bf16, int32_t pos0, int32_t ring_mask,
                                 int32_t M, hipStream_t stream) {
    if (M <= 0) return;
    dim3 grid(5, static_cast<uint32_t>(M), 1);
    dim3 block(256, 1, 1);
    nvfp4_mtp_ring_write_k<<<grid, block, 0, stream>>>(
        static_cast<const uint16_t*>(src_x_bf16),
        static_cast<uint16_t*>(dst_ring_bf16),
        pos0, ring_mask);
}

void launch_nvfp4_mtp_emb_rms_norm(const int32_t* d_tok_ids, const void* d_embed_tokens, const void* d_norm_w,
                                   void* d_out_bf16, int32_t M, float eps, hipStream_t stream) {
    if (M <= 0) return;
    nvfp4_mtp_emb_rms_norm_k<<<static_cast<uint32_t>(M), 256, 0, stream>>>(
        d_tok_ids,
        static_cast<const uint16_t*>(d_embed_tokens),
        static_cast<const uint16_t*>(d_norm_w),
        static_cast<uint16_t*>(d_out_bf16),
        eps);
}

void launch_nvfp4_mtp_hidden_rms_norm(const void* d_r_base, int32_t ring_pos0, int32_t ring_mask,
                                      const void* d_norm_w, void* d_out_bf16, int32_t M, float eps,
                                      bool per_stream, hipStream_t stream) {
    if (M <= 0) return;
    nvfp4_mtp_hidden_rms_norm_k<<<static_cast<uint32_t>(M), 256, 0, stream>>>(
        static_cast<const uint16_t*>(d_r_base),
        ring_pos0, ring_mask,
        static_cast<const uint16_t*>(d_norm_w),
        static_cast<uint16_t*>(d_out_bf16),
        eps, per_stream ? 1 : 0);
}

void launch_nvfp4_mtp_stream_broadcast_add(const void* d_h2_bf16, const void* d_e2_bf16, void* d_r_out_bf16,
                                           int32_t M, hipStream_t stream) {
    if (M <= 0) return;
    dim3 grid(40, static_cast<uint32_t>(M), 1);
    dim3 block(256, 1, 1);
    nvfp4_mtp_stream_broadcast_add_k<<<grid, block, 0, stream>>>(
        static_cast<const uint16_t*>(d_h2_bf16),
        static_cast<const uint16_t*>(d_e2_bf16),
        static_cast<uint16_t*>(d_r_out_bf16));
}

void launch_nvfp4_mtp_kv_idx_proj(const void* d_x_bf16, const void* d_w_kv_idx_bf16,
                                  void* d_k_out_bf16, void* d_v_out_bf16, void* d_idx_qk_out_bf16,
                                  int32_t M, hipStream_t stream) {
    if (M <= 0) return;
    dim3 grid(208, static_cast<uint32_t>(M), 1);
    dim3 block(256, 1, 1);
    nvfp4_mtp_kv_idx_proj_k<<<grid, block, 0, stream>>>(
        static_cast<const uint16_t*>(d_x_bf16),
        static_cast<const uint16_t*>(d_w_kv_idx_bf16),
        static_cast<uint16_t*>(d_k_out_bf16),
        static_cast<uint16_t*>(d_v_out_bf16),
        static_cast<uint16_t*>(d_idx_qk_out_bf16));
}

void launch_nvfp4_mtp_moe_decode_gate_up(const void* d_mixed_bf16, const int32_t* d_cur_ids, const int64_t* d_ptrs,
                                         void* d_moe_inter_bf16, hipStream_t stream) {
    dim3 grid(80, 10, 1);
    dim3 block(256, 1, 1);
    nvfp4_mtp_moe_decode_gate_up_k<<<grid, block, 0, stream>>>(
        static_cast<const uint16_t*>(d_mixed_bf16),
        d_cur_ids, d_ptrs,
        static_cast<uint16_t*>(d_moe_inter_bf16));
}

void launch_nvfp4_mtp_moe_decode_down(const void* d_moe_inter_bf16, const int32_t* d_cur_ids, const int64_t* d_ptrs,
                                      void* d_moe_y_bf16, hipStream_t stream) {
    dim3 grid(320, 10, 1);
    dim3 block(256, 1, 1);
    nvfp4_mtp_moe_decode_down_k<<<grid, block, 0, stream>>>(
        static_cast<const uint16_t*>(d_moe_inter_bf16),
        d_cur_ids, d_ptrs,
        static_cast<uint16_t*>(d_moe_y_bf16));
}

void launch_nvfp4_argmax_prob_margin_248320(const void* d_logits_bf16, int32_t* d_out_tok,
                                            Nvfp4ArgmaxSummary* d_out_summary, int32_t M,
                                            hipStream_t stream) {
    if (M <= 0) return;
    nvfp4_argmax_prob_margin_248320_k<<<static_cast<uint32_t>(M), 1024, 0, stream>>>(
        static_cast<const uint16_t*>(d_logits_bf16),
        d_out_tok,
        d_out_summary);
}

void launch_nvfp4_qsa_pin_sink_and_window(float* d_scores, int32_t pos0, int32_t M, int32_t score_stride,
                                          int32_t n_sink, int32_t n_local, hipStream_t stream) {
    if (M <= 0) return;
    dim3 grid(1, static_cast<uint32_t>(M), 1);
    dim3 block(128, 1, 1);
    nvfp4_qsa_pin_sink_and_window_k<<<grid, block, 0, stream>>>(
        d_scores, pos0, score_stride, n_sink, n_local);
}

void launch_nvfp4_moe_cap_gpu_cold(int32_t* d_cur_ids, int32_t* d_gpu_align_ids, const int64_t* d_ptrs_decode,
                                   const int64_t* d_ptrs_vram, int32_t pairs, int32_t max_gpu_cold,
                                   hipStream_t stream) {
    if (pairs <= 0) return;
    nvfp4_moe_cap_gpu_cold_k<<<1, 32, 0, stream>>>(
        d_cur_ids, d_gpu_align_ids, d_ptrs_decode, d_ptrs_vram, pairs, max_gpu_cold);
}

void launch_nvfp4_mtp_dvocab_gemv(const void* d_x_bf16, const void* d_lm_head_bf16, const int32_t* d_dvocab,
                                  void* d_logits_out_bf16, int32_t n_dvocab, hipStream_t stream) {
    if (n_dvocab <= 0) return;
    uint32_t blocks = static_cast<uint32_t>((n_dvocab + 7) / 8);
    nvfp4_mtp_dvocab_gemv_k<<<blocks, 256, 0, stream>>>(
        static_cast<const uint16_t*>(d_x_bf16),
        static_cast<const uint16_t*>(d_lm_head_bf16),
        d_dvocab,
        static_cast<uint16_t*>(d_logits_out_bf16),
        n_dvocab);
}

void launch_nvfp4_argmax_prob_margin_dvocab(const void* d_logits_bf16, const int32_t* d_dvocab, int32_t n_dvocab,
                                            int32_t* d_out_tok, Nvfp4ArgmaxSummary* d_out_summary,
                                            hipStream_t stream) {
    if (n_dvocab <= 0) return;
    nvfp4_argmax_prob_margin_dvocab_k<<<1, 1024, 0, stream>>>(
        static_cast<const uint16_t*>(d_logits_bf16),
        d_dvocab, n_dvocab, d_out_tok, d_out_summary);
}

void launch_nvfp4_mtp_topk_truncate(int32_t* d_cur_ids, float* d_cur_wts, int32_t keep_k,
                                    hipStream_t stream) {
    if (keep_k <= 0 || keep_k >= 10) return;
    nvfp4_mtp_topk_truncate_k<<<1, 32, 0, stream>>>(d_cur_ids, d_cur_wts, keep_k);
}

void launch_nvfp4_qsa_block_keys_from_batch(void* d_raw_k_cache, const void* d_idx_qk_b, const void* d_idx_k_ln,
                                            const void* d_rope_cos_bf16, const void* d_rope_sin_bf16,
                                            void* d_block_k_cache, int64_t bk_host_delta_elems,
                                            int32_t pos0, int32_t M, int32_t b_start, int32_t b_cnt,
                                            int32_t vram_blocks, float eps, hipStream_t stream) {
    if (b_cnt <= 0 && M <= 0) return;
    uint32_t grid_x = static_cast<uint32_t>(std::max(1, (b_cnt + 3) / 4));
    nvfp4_qsa_block_keys_from_batch_k<<<grid_x, 128, 0, stream>>>(
        static_cast<uint16_t*>(d_raw_k_cache),
        static_cast<const uint16_t*>(d_idx_qk_b),
        static_cast<const uint16_t*>(d_idx_k_ln),
        static_cast<const uint16_t*>(d_rope_cos_bf16),
        static_cast<const uint16_t*>(d_rope_sin_bf16),
        static_cast<uint16_t*>(d_block_k_cache),
        bk_host_delta_elems, pos0, M, b_start, b_cnt, vram_blocks, eps);
}

void launch_nvfp4_qsa_score_blocks_batched(const void* d_idx_qk_b, const void* d_idx_q_ln,
                                           const void* d_rope_cos_bf16, const void* d_rope_sin_bf16,
                                           const void* d_block_k_cache, float* d_scores,
                                           int64_t bk_host_delta_elems, int32_t pos_cur, int32_t m_cur,
                                           int32_t vram_blocks, int32_t score_stride,
                                           int32_t n_sink, int32_t n_local, float eps,
                                           hipStream_t stream) {
    if (m_cur <= 0) return;
    int32_t max_nb = (pos_cur + m_cur) / 4;
    uint32_t score_grid = static_cast<uint32_t>(std::max(1, (max_nb + 255) / 256));
    uint32_t tok_tiles = static_cast<uint32_t>((m_cur + 7) / 8);
    dim3 grid(score_grid, tok_tiles, 1);
    dim3 block(256, 1, 1);
    nvfp4_qsa_score_blocks_batched_k<<<grid, block, 0, stream>>>(
        static_cast<const uint16_t*>(d_idx_qk_b),
        static_cast<const uint16_t*>(d_idx_q_ln),
        static_cast<const uint16_t*>(d_rope_cos_bf16),
        static_cast<const uint16_t*>(d_rope_sin_bf16),
        static_cast<const uint16_t*>(d_block_k_cache),
        d_scores, bk_host_delta_elems, pos_cur, m_cur,
        vram_blocks, score_stride, n_sink, n_local, eps);
}

}  // namespace strata::kernels
