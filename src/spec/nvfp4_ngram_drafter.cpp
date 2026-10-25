// src/spec/nvfp4_ngram_drafter.cpp - Multi-level consensus suffix drafter with rejection memory.
#include "strata/spec/nvfp4_ngram_drafter.hpp"

#include <cctype>

namespace strata::spec {

Nvfp4NgramDrafter::Nvfp4NgramDrafter(const strata::artifact::QwenTokenizer& tokenizer)
    : tokenizer_(tokenizer) {}

void Nvfp4NgramDrafter::clear_rejections() {
    rejected_set_.clear();
}

uint64_t Nvfp4NgramDrafter::make_rej_key(int32_t p2, int32_t p1, int32_t cand) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(p2) & 0x1FFFFFu) << 42) |
           (static_cast<uint64_t>(static_cast<uint32_t>(p1) & 0x1FFFFFu) << 21) |
           static_cast<uint64_t>(static_cast<uint32_t>(cand) & 0x1FFFFFu);
}

void Nvfp4NgramDrafter::record_rejection(int32_t prev2_t, int32_t prev_t, int32_t bad_cand) {
    rejected_set_[make_rej_key(prev2_t, prev_t, bad_cand)] = 1;
    rejected_set_[make_rej_key(-1, prev_t, bad_cand)] = 1;
}

bool Nvfp4NgramDrafter::is_rejected(int32_t p2, int32_t p1, int32_t cand) const {
    if (rejected_set_.find(make_rej_key(p2, p1, cand)) != rejected_set_.end()) return true;
    if (p2 == -1 && rejected_set_.find(make_rej_key(-1, p1, cand)) != rejected_set_.end()) return true;
    return false;
}

const std::string& Nvfp4NgramDrafter::get_tok_str(int32_t id) {
    auto it = tok_str_cache_.find(id);
    if (it != tok_str_cache_.end()) return it->second;
    return tok_str_cache_.emplace(id, tokenizer_.decode_token(id)).first->second;
}

bool Nvfp4NgramDrafter::is_clause_break(int32_t cand_id, int32_t next_id) {
    if (cand_id < 0 || cand_id >= 248040) return true;
    const std::string& s = get_tok_str(cand_id);
    if (s.empty()) return true;
    char c0 = s[0];
    if (c0 == '\n' || c0 == '\r' || c0 == ',' || c0 == ';' || c0 == ')' || c0 == ']' || c0 == '}') return true;
    if (c0 == '.') {
        if (s.size() == 1 && next_id >= 0 && next_id < 248040) {
            const std::string& ns = get_tok_str(next_id);
            if (!ns.empty() && std::isdigit(static_cast<unsigned char>(ns[0]))) return false;
        }
        return true;
    }
    return false;
}

bool Nvfp4NgramDrafter::is_subword_cont(int32_t prev_id, int32_t cand_id, int32_t next_id) {
    if (prev_id < 0 || cand_id < 0 || cand_id >= 248040) return false;
    const std::string& ps = get_tok_str(prev_id);
    const std::string& cs = get_tok_str(cand_id);
    if (ps.empty() || cs.empty()) return false;
    char pb = ps.back();
    if (!std::isalnum(static_cast<unsigned char>(pb)) && pb != '-') return false;
    char c0 = cs[0];
    if (std::isalnum(static_cast<unsigned char>(c0))) return true;
    if ((c0 == '-' || c0 == '/') && cs.size() >= 2 && std::isalnum(static_cast<unsigned char>(cs[1]))) return true;
    if ((cs == "-" || cs == ".") && next_id >= 0 && next_id < 248040) {
        const std::string& ns = get_tok_str(next_id);
        if (!ns.empty() && std::isalnum(static_cast<unsigned char>(ns[0]))) return true;
    }
    return false;
}

bool Nvfp4NgramDrafter::is_digit_tok(int32_t id) {
    if (id < 0 || id >= 248040) return false;
    const std::string& s = get_tok_str(id);
    size_t pos = (!s.empty() && s[0] == ' ') ? 1 : 0;
    return pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]));
}

bool Nvfp4NgramDrafter::is_entity_or_num(int32_t id) {
    if (id < 0 || id >= 248040) return false;
    const std::string& s = get_tok_str(id);
    if (s.empty()) return false;
    size_t pos = (s[0] == ' ') ? 1 : 0;
    if (s.size() - pos < 2) return false;
    unsigned char c0 = static_cast<unsigned char>(s[pos]);
    if (std::isdigit(c0)) return true;
    if (!std::isupper(c0)) return false;
    std::string w = s.substr(pos);
    static const char* const kCapStop[] = {
        "The", "A", "An", "In", "On", "At", "To", "For", "Of", "With", "By", "From", "As",
        "Is", "Are", "Was", "Were", "Be", "This", "That", "These", "Those", "It", "Its",
        "They", "We", "You", "And", "Or", "But", "If", "When", "While", "Where", "Why",
        "How", "What", "Which", "Who", "Here", "There", "Each", "Both", "All", "Some",
        "No", "Not", "Only", "Also", "However", "Therefore", "Thus", "Summarize",
        "Explain", "Describe", "List", "Compare", "Key", "Summary", "Overview", "Note",
        "Contains", "Equipped", "Uses", "Based", "Running", "Routes", "Total", "Step"
    };
    for (const char* sw : kCapStop) {
        if (w == sw) return false;
    }
    return true;
}

bool Nvfp4NgramDrafter::is_new_number_head(int32_t /*prev2_id*/, int32_t prev_id, int32_t cand_id) {
    if (!is_digit_tok(cand_id) || prev_id < 0) return false;
    const std::string& ps = get_tok_str(prev_id);
    if (ps.empty()) return false;
    char pb = ps.back();
    return (pb == '(' || pb == '*' || pb == '[');
}

bool Nvfp4NgramDrafter::is_content_token(int32_t id) {
    if (is_entity_or_num(id)) return true;
    if (id < 0 || id >= 248040) return false;
    const std::string& s = get_tok_str(id);
    size_t pos = (!s.empty() && s[0] == ' ') ? 1 : 0;
    if (s.size() - pos < 4) return false;
    if (!std::isalpha(static_cast<unsigned char>(s[pos]))) return false;
    std::string w = s.substr(pos);
    static const char* const kLowStop[] = {
        "that", "this", "with", "from", "have", "will", "what", "when", "where", "which",
        "while", "their", "there", "these", "those", "about", "into", "over", "after",
        "before", "under", "between", "through", "during", "without", "within", "because",
        "should", "would", "could", "other", "such", "more", "most", "some", "many",
        "much", "very", "also", "only", "even", "just", "than", "then", "them", "they",
        "were", "been", "being", "does", "make", "made", "used", "using", "uses", "each", "both"
    };
    for (const char* sw : kLowStop) {
        if (w == sw) return false;
    }
    return true;
}

void Nvfp4NgramDrafter::draft(int32_t cur_tok, size_t prompt_len, const std::vector<int32_t>& history,
                              bool enabled, std::vector<int32_t>& draft_out) {
    draft_out.clear();
    if (!enabled || prompt_len < 8 || history.size() < 4) return;
    const size_t n = history.size();
    const int32_t w0 = history[n - 3];
    const int32_t w1 = history[n - 2];
    const int32_t w2 = history[n - 1];
    const int32_t w3 = cur_tok;

    // Multi-level consensus suffix matcher: L = 4, 3, 2, 1
    for (int L = 4; L >= 1; --L) {
        int total1 = 0, total2 = 0, total3 = 0, total4 = 0;
        int agree1 = 0, agree2 = 0, agree3 = 0, agree4 = 0;
        int32_t c1 = -1, c2 = -1, c3 = -1, c4 = -1;
        const size_t min_i = static_cast<size_t>(L - 1);
        for (size_t i = n - 2;; --i) {
            bool match = (history[i] == w3);
            if (match && L >= 2) match = (history[i - 1] == w2);
            if (match && L >= 3) match = (history[i - 2] == w1);
            if (match && L >= 4) match = (history[i - 3] == w0);
            if (match && (i + 1) < n) {
                int32_t cand1 = history[i + 1];
                if (cand1 >= 0 && cand1 < 248040 && !is_rejected(w2, w3, cand1) &&
                    !(L == 1 && is_rejected(-1, w3, cand1))) {
                    total1++;
                    int32_t cand2 = ((i + 2) < n && history[i + 2] < 248040) ? history[i + 2] : -1;
                    int32_t cand3 = (cand2 >= 0 && (i + 3) < n && history[i + 3] < 248040) ? history[i + 3] : -1;
                    int32_t cand4 = (cand3 >= 0 && (i + 4) < n && history[i + 4] < 248040) ? history[i + 4] : -1;
                    if (c1 < 0) {
                        c1 = cand1;
                        c2 = cand2;
                        c3 = cand3;
                        c4 = cand4;
                    }
                    if (cand1 == c1) {
                        agree1++;
                        if (cand2 >= 0) {
                            if (c2 < 0) c2 = cand2;
                            total2++;
                            if (cand2 == c2) {
                                agree2++;
                                if (cand3 >= 0) {
                                    if (c3 < 0) c3 = cand3;
                                    total3++;
                                    if (cand3 == c3) {
                                        agree3++;
                                        if (cand4 >= 0) {
                                            if (c4 < 0) c4 = cand4;
                                            total4++;
                                            if (cand4 == c4) agree4++;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if (i == min_i) break;
        }
        if (total1 == 0 || c1 < 0) continue;

        auto space_ok = [&](int32_t ck, int32_t ck_next, int ag_next, int tot_next, int32_t prev_t) -> bool {
            if (get_tok_str(ck) != " ") return true;
            return (ck_next >= 0 && tot_next > 0 && ag_next == tot_next && !is_new_number_head(prev_t, ck, ck_next));
        };

        if (L == 4) {
            if (is_new_number_head(w2, w3, c1) && agree1 < 2) continue;
            draft_out.push_back(c1);
            if (c2 >= 0 && !is_rejected(w3, c1, c2) && !is_new_number_head(w3, c1, c2)) {
                draft_out.push_back(c2);
                if (c3 >= 0 && !is_rejected(c1, c2, c3) && !is_new_number_head(c1, c2, c3)) {
                    draft_out.push_back(c3);
                }
            }
            if (draft_out.size() >= 2) return;
            draft_out.clear();
            continue;
        }
        if (L == 3) {
            if (agree1 == total1 && !is_clause_break(c1, c2) && !is_new_number_head(w2, w3, c1) &&
                space_ok(c1, c2, agree2, total2, w3)) {
                draft_out.push_back(c1);
                if (c2 >= 0 && total2 > 0 && agree2 == total2 && !is_rejected(w3, c1, c2) &&
                    !is_clause_break(c2, c3) && !is_new_number_head(w3, c1, c2) &&
                    space_ok(c2, c3, agree3, total3, c1)) {
                    draft_out.push_back(c2);
                    if (c3 >= 0 && total3 > 0 && agree3 == total3 && !is_rejected(c1, c2, c3) &&
                        !is_clause_break(c3, c4) && !is_new_number_head(c1, c2, c3) &&
                        space_ok(c3, c4, agree4, total4, c2)) {
                        draft_out.push_back(c3);
                    }
                }
                if (draft_out.size() >= 2) return;
                draft_out.clear();
            }
        }
        if (L == 2) {
            bool ok2 = (agree1 == total1) && !is_clause_break(c1, c2) && !is_new_number_head(w2, w3, c1) &&
                       space_ok(c1, c2, agree2, total2, w3) &&
                       (is_subword_cont(w3, c1, c2) || is_subword_cont(w2, w3, c1) || is_content_token(w3) ||
                        (is_content_token(w2) && is_content_token(c1)));
            if (ok2) {
                draft_out.push_back(c1);
                if (c2 >= 0 && total2 > 0 && agree2 == total2 && !is_rejected(w3, c1, c2) &&
                    !is_clause_break(c2, c3) && !is_new_number_head(w3, c1, c2) &&
                    space_ok(c2, c3, agree3, total3, c1) &&
                    (is_subword_cont(c1, c2, c3) || is_content_token(c2) || total1 >= 2)) {
                    draft_out.push_back(c2);
                    if (c3 >= 0 && total3 > 0 && agree3 == total3 && !is_rejected(c1, c2, c3) &&
                        !is_clause_break(c3, c4) && !is_new_number_head(c1, c2, c3) &&
                        space_ok(c3, c4, agree3, total4, c2) &&
                        (is_subword_cont(c2, c3, c4) || is_entity_or_num(c3) || total1 >= 2)) {
                        draft_out.push_back(c3);
                    }
                }
                if (draft_out.size() >= 2) return;
                draft_out.clear();
            }
        }
        if (L == 1) {
            const std::string& w3_s = get_tok_str(w3);
            size_t w3_pos = (!w3_s.empty() && w3_s[0] == ' ') ? 1 : 0;
            bool w3_substantial = (w3_s.size() - w3_pos >= 2) || is_entity_or_num(w3);
            bool sub1 = w3_substantial && is_subword_cont(w3, c1, c2);
            bool ent1 = (total1 <= 6) && is_entity_or_num(w3) &&
                        (is_entity_or_num(c1) || (get_tok_str(c1) == " " && space_ok(c1, c2, agree2, total2, w3)));
            if (agree1 == total1 && !is_clause_break(c1, c2) && !is_new_number_head(w2, w3, c1) &&
                space_ok(c1, c2, agree2, total2, w3) && (sub1 || ent1)) {
                draft_out.push_back(c1);
                if (c2 >= 0 && total2 > 0 && agree2 == total2 && !is_rejected(w3, c1, c2) &&
                    !is_clause_break(c2, c3) && !is_new_number_head(w3, c1, c2) &&
                    space_ok(c2, c3, agree3, total3, c1) &&
                    (is_subword_cont(c1, c2, c3) || is_entity_or_num(c2) ||
                     (get_tok_str(c1) == " " && is_digit_tok(c2)))) {
                    draft_out.push_back(c2);
                    if (c3 >= 0 && total3 > 0 && agree3 == total3 && !is_rejected(c1, c2, c3) &&
                        !is_clause_break(c3, c4) && !is_new_number_head(c1, c2, c3) &&
                        space_ok(c3, c4, agree4, total4, c2) &&
                        (is_subword_cont(c2, c3, c4) || (is_digit_tok(c2) && is_digit_tok(c3)))) {
                        draft_out.push_back(c3);
                    }
                }
                if (draft_out.size() >= 2) return;
                draft_out.clear();
            }
        }
    }
}

}  // namespace strata::spec
