// include/strata/kernels/nvfp4_mtp_kernels.hpp - RDNA4 (gfx1201) HIP kernels for native Multi-Token Prediction (MTP),
// target logit confidence/margin extraction, NSA/CSA sink+window pinning, and capped hybrid small-batch MoE routing.
#pragma once

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

// Per-expert byte layout for the 512 FP8 MTP experts (mtp.layers.0.mlp.experts.{0..511}):
//   gate_proj.weight [640, 2560] F8_E4M3 (1,638,400 B) + weight_scale_inv [5, 20] BF16 (200 B, padded to 256 B)
//   up_proj.weight   [640, 2560] F8_E4M3 (1,638,400 B) + weight_scale_inv [5, 20] BF16 (200 B, padded to 256 B)
//   down_proj.weight [2560, 640] F8_E4M3 (1,638,400 B) + weight_scale_inv [20, 5] BF16 (200 B, padded to 576 B)
inline constexpr size_t MTP_EXP_W_BYTES = 1638400;
inline constexpr size_t MTP_EXP_S_BYTES = 200;
inline constexpr size_t MTP_OFF_GATE_W = 0;
inline constexpr size_t MTP_OFF_UP_W = 1638400;
inline constexpr size_t MTP_OFF_DOWN_W = 3276800;
inline constexpr size_t MTP_OFF_GATE_S = 4915200;
inline constexpr size_t MTP_OFF_UP_S = 4915456;
inline constexpr size_t MTP_OFF_DOWN_S = 4915712;
inline constexpr size_t MTP_EXP_STRIDE_BYTES = 4916288;

struct Nvfp4ArgmaxSummary {
    int32_t tok_id;
    float prob;
    float margin;
};

// Write M rows of BF16 [M, 10240] final 4-stream residuals into the 512-slot ring buffer at positions [pos0, pos0 + M).
void launch_nvfp4_mtp_ring_write(const void* src_x_bf16, void* dst_ring_bf16, int32_t pos0, int32_t ring_mask,
                                 int32_t M, hipStream_t stream = nullptr);

// Gather M token embeddings from embed_tokens [248320, 2560] and apply (1 + w) RMSNorm over 2560 elements.
void launch_nvfp4_mtp_emb_rms_norm(const int32_t* d_tok_ids, const void* d_embed_tokens, const void* d_norm_w,
                                   void* d_out_bf16, int32_t M, float eps, hipStream_t stream = nullptr);

// Apply (1 + w) RMSNorm over 10240 elements per row (reading row t from d_r_base[((ring_pos0 + t) & ring_mask) * 10240]).
void launch_nvfp4_mtp_hidden_rms_norm(const void* d_r_base, int32_t ring_pos0, int32_t ring_mask,
                                      const void* d_norm_w, void* d_out_bf16, int32_t M, float eps,
                                      bool per_stream = false, hipStream_t stream = nullptr);

// Broadcast-add e2 [M, 2560] to each of the 4 streams of h2 [M, 4, 2560] -> r_out [M, 4, 2560].
void launch_nvfp4_mtp_stream_broadcast_add(const void* d_h2_bf16, const void* d_e2_bf16, void* d_r_out_bf16,
                                           int32_t M, hipStream_t stream = nullptr);

// Compute K/V + indexer QK projections for M tokens: x [M, 2560] x w_kv_idx [1664, 2560]
// -> k_out [M, 512], v_out [M, 512], idx_qk_out [M, 640].
void launch_nvfp4_mtp_kv_idx_proj(const void* d_x_bf16, const void* d_w_kv_idx_bf16,
                                  void* d_k_out_bf16, void* d_v_out_bf16, void* d_idx_qk_out_bf16,
                                  int32_t M, hipStream_t stream = nullptr);

// Execute GPU-cold FP8 MTP experts (gate_proj + up_proj + SwiGLU) for decode (M = 1, top-10 routed slots).
void launch_nvfp4_mtp_moe_decode_gate_up(const void* d_mixed_bf16, const int32_t* d_cur_ids, const int64_t* d_ptrs,
                                         void* d_moe_inter_bf16, hipStream_t stream = nullptr);

// Execute GPU-cold FP8 MTP experts (down_proj) for decode (M = 1, top-10 routed slots).
void launch_nvfp4_mtp_moe_decode_down(const void* d_moe_inter_bf16, const int32_t* d_cur_ids, const int64_t* d_ptrs,
                                      void* d_moe_y_bf16, hipStream_t stream = nullptr);

// Single-pass argmax + exact full-vocab softmax probability + top-1/top-2 logit margin over [M, 248320] BF16 logits.
void launch_nvfp4_argmax_prob_margin_248320(const void* d_logits_bf16, int32_t* d_out_tok,
                                            Nvfp4ArgmaxSummary* d_out_summary, int32_t M,
                                            hipStream_t stream = nullptr);

// Pin initial attention sink blocks (0..n_sink-1) and local sliding-window blocks ([nb - n_local, nb)) in QSA block scores.
void launch_nvfp4_qsa_pin_sink_and_window(float* d_scores, int32_t pos0, int32_t M, int32_t score_stride,
                                          int32_t n_sink, int32_t n_local, hipStream_t stream = nullptr);

// Cap GPU-cold PCIe experts to max_gpu_cold in small-batch verification (M <= 4), marking excess cold experts with 0x10000
// for parallel execution on the 16-core AVX-512 CPU pool and producing a GPU-filtered ID list for moe_align.
void launch_nvfp4_moe_cap_gpu_cold(int32_t* d_cur_ids, int32_t* d_gpu_align_ids, const int64_t* d_ptrs_decode,
                                   const int64_t* d_ptrs_vram, int32_t pairs, int32_t max_gpu_cold,
                                   hipStream_t stream = nullptr);

}  // namespace strata::kernels
