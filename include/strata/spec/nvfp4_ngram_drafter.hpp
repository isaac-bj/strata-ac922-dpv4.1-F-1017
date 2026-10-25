// include/strata/spec/nvfp4_ngram_drafter.hpp - Multi-level consensus suffix drafter with rejection memory.
#pragma once

#include "strata/artifact/qwen_tokenizer.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::spec {

class Nvfp4NgramDrafter {
public:
    explicit Nvfp4NgramDrafter(const strata::artifact::QwenTokenizer& tokenizer);

    void clear_rejections();
    void record_rejection(int32_t prev2_t, int32_t prev_t, int32_t bad_cand);

    void draft(int32_t cur_tok, size_t prompt_len, const std::vector<int32_t>& history,
               bool enabled, std::vector<int32_t>& draft_out);

private:
    static uint64_t make_rej_key(int32_t p2, int32_t p1, int32_t cand);
    bool is_rejected(int32_t p2, int32_t p1, int32_t cand) const;
    const std::string& get_tok_str(int32_t id);

    bool is_clause_break(int32_t cand_id, int32_t next_id);
    bool is_subword_cont(int32_t prev_id, int32_t cand_id, int32_t next_id);
    bool is_digit_tok(int32_t id);
    bool is_entity_or_num(int32_t id);
    bool is_new_number_head(int32_t prev2_id, int32_t prev_id, int32_t cand_id);
    bool is_content_token(int32_t id);

    const strata::artifact::QwenTokenizer& tokenizer_;
    std::unordered_map<uint64_t, uint8_t> rejected_set_;
    std::unordered_map<int32_t, std::string> tok_str_cache_;
};

}  // namespace strata::spec
