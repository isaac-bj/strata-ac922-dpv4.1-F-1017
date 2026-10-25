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

}  // namespace strata::kernels
