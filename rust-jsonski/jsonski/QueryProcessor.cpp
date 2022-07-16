#include "QueryProcessor.h"
#include <immintrin.h>

#include <emmintrin.h>
#include <string.h>

#include <sys/time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <pthread.h>
#include <malloc.h>
#include <sys/time.h>
#include <sys/file.h>
#include <unistd.h>
#include <sched.h>
#include <unordered_map>

using namespace std;

QueryProcessor::QueryProcessor(string query) {
    JSONPathParser::updateQueryAutomaton(query, this->qa);
    this->mNumMatches = 0;
    this->mText = new char[MAX_TEXT_LENGTH];
    init(); 
}

void QueryProcessor::init() {
    structural_table =
        _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '{', 0, '}', 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '{', 0, '}', 0, 0);
    struct_mask = _mm256_set1_epi8(0x20);
    // vectors for structural characters
    v_quote = _mm256_set1_epi8(0x22);
    v_colon = _mm256_set1_epi8(0x3a);
    v_escape = _mm256_set1_epi8(0x5c);
    v_lbrace = _mm256_set1_epi8(0x7b);
    v_rbrace = _mm256_set1_epi8(0x7d);
    v_comma = _mm256_set1_epi8(0x2c);
    v_lbracket = _mm256_set1_epi8(0x5b);
    v_rbracket = _mm256_set1_epi8(0x5d);
    // some global variables among internal functions
    top_word = -1;
    prev_iter_ends_odd_backslash = 0ULL;
    prev_iter_inside_quote = 0ULL;
    even_bits = 0x5555555555555555ULL;
    odd_bits = ~even_bits;
    start_id = 0;
    cb_mask = 0, colon_mask = 0, comma_mask = 0; mask = 0;
    colonbit = 0; quotebit = 0; commabit = 0; bracketbit = 0;
    cur_word = false;
    top_word = -1;
    cur_pos = 0; 
}

QueryProcessor::~QueryProcessor()
{
    if (mText) {
        free(mText);
        mText = NULL;
    }
}

void QueryProcessor::setRecordText(char* rec_text, long length) {
    this->mRecord = rec_text;
    this->mRecordLength = length;
    this->mNumTmpWords = length / 32;
    this->mNumWords = length / 64; 
}

// build quote bitmap and string mask bitmap for the current word
__attribute__((always_inline)) void QueryProcessor::build_bitmap_basic() {
    unsigned long quotebit0, escapebit0;
    unsigned long quotebit, escapebit;
    // step 1: build structural quote and escape bitmaps for the current word
    // first half of bitmap
    top_word = start_id / 2; // word id 
    unsigned long i = start_id * 32;
    v_text0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(mRecord + i));
    quotebit0 = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v_text0, v_quote));
    escapebit0 = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v_text0, v_escape));
    // second half of bitmap 
    ++start_id;
    i = (start_id) * 32;
    v_text = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(mRecord + i));
    quotebit = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v_text, v_quote));
    escapebit = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v_text, v_escape));
    quotebit = (quotebit << 32) | quotebit0;
    escapebit = (escapebit << 32) | escapebit0;
    // step 2: update structural quote bitmaps
    uint64_t bs_bits = escapebit;
    uint64_t start_edges = bs_bits & ~(bs_bits << 1);
    int64_t even_start_mask = even_bits ^ prev_iter_ends_odd_backslash;
    uint64_t even_starts = start_edges & even_start_mask;
    uint64_t odd_starts = start_edges & ~even_start_mask;
    uint64_t even_carries = bs_bits + even_starts;
    int64_t odd_carries;
    bool iter_ends_odd_backslash = __builtin_uaddll_overflow(bs_bits, odd_starts,
        (unsigned long long *)(&odd_carries));
    odd_carries |= prev_iter_ends_odd_backslash;
    prev_iter_ends_odd_backslash = iter_ends_odd_backslash ? 0x1ULL : 0x0ULL;
    uint64_t even_carry_ends = even_carries & ~bs_bits;
    uint64_t odd_carry_ends = odd_carries & ~bs_bits;
    uint64_t even_start_odd_end = even_carry_ends & odd_bits;
    uint64_t odd_start_even_end = odd_carry_ends & even_bits;
    uint64_t odd_ends = even_start_odd_end | odd_start_even_end;
    quote_bits = quotebit & ~odd_ends;
     // step 3: build string mask bitmaps
    str_mask = _mm_cvtsi128_si64(_mm_clmulepi64_si128(
        _mm_set_epi64x(0ULL, quote_bits), _mm_set1_epi8(0xFFu), 0));
    str_mask ^= prev_iter_inside_quote;
    prev_iter_inside_quote = static_cast<uint64_t>(static_cast<int64_t>(str_mask) >> 63);
}

__attribute__((always_inline)) void QueryProcessor::build_bitmap_colon(bitmap& bm) {
    unsigned long colonbit0, colonbit;
    colonbit0 = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v_text0, v_colon));
    colonbit = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v_text, v_colon));
    bm.colonbit = (colonbit << 32) | colonbit0;
    bm.colonbit = bm.colonbit & (~str_mask);
}

__attribute__((always_inline)) void QueryProcessor::get_bitmap_colon(bitmap& bm) {
    if (bm.has_colon == false) {
        build_bitmap_colon(bm);
        bm.has_colon = true;
    }
}

__attribute__((always_inline)) void QueryProcessor::build_bitmap_comma(bitmap& bm) {
    unsigned long commabit0, commabit;
    commabit0 = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v_text0, v_comma));
    commabit = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v_text, v_comma));
    bm.commabit = (commabit << 32) | commabit0;
    bm.commabit = bm.commabit & (~str_mask);
}

__attribute__((always_inline)) void QueryProcessor::get_bitmap_comma(bitmap& bm) {
    if (bm.has_comma == false) {
        build_bitmap_comma(bm);
        bm.has_comma = true;
    }
}

__attribute__((always_inline)) void QueryProcessor::build_bitmap_lbrace(bitmap& bm) {
    unsigned long lbracebit0, lbracebit;
    lbracebit0 = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v_text0, v_lbrace));
    lbracebit = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v_text, v_lbrace));
    bm.lbracebit = (lbracebit << 32) | lbracebit0;
    bm.lbracebit = bm.lbracebit & (~str_mask);
}

__attribute__((always_inline)) void QueryProcessor::get_bitmap_lbrace(bitmap& bm) {
    if (bm.has_lbrace == false) {
        build_bitmap_lbrace(bm);
        bm.has_lbrace = true;
    }
}

__attribute__((always_inline)) void QueryProcessor::build_bitmap_rbrace(bitmap& bm) {
    unsigned long rbracebit0, rbracebit;
    rbracebit0 = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v_text0, v_rbrace));
    rbracebit = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v_text, v_rbrace));
    bm.rbracebit = (rbracebit << 32) | rbracebit0;
    bm.rbracebit = bm.rbracebit & (~str_mask);
}

__attribute__((always_inline)) void QueryProcessor::get_bitmap_rbrace(bitmap& bm) {
    if (bm.has_rbrace == false) {
        build_bitmap_rbrace(bm);
        bm.has_rbrace = true;
    }
}

__attribute__((always_inline)) void QueryProcessor::build_bitmap_lbracket(bitmap& bm) {
    unsigned long lbracketbit0, lbracketbit;
    lbracketbit0 = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v_text0, v_lbracket));
    lbracketbit = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v_text, v_lbracket));
    bm.lbracketbit = (lbracketbit << 32) | lbracketbit0;
    bm.lbracketbit = bm.lbracketbit & (~str_mask);
}

__attribute__((always_inline)) void QueryProcessor::get_bitmap_lbracket(bitmap& bm) {
    if (bm.has_lbracket == false) {
        build_bitmap_lbracket(bm);
        bm.has_lbracket = true;
    }
}

__attribute__((always_inline)) void QueryProcessor::build_bitmap_rbracket(bitmap& bm) {
    unsigned long rbracketbit0, rbracketbit;
    rbracketbit0 = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v_text0, v_rbracket));
    rbracketbit = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v_text, v_rbracket));
    bm.rbracketbit = (rbracketbit << 32) | rbracketbit0;
    bm.rbracketbit = bm.rbracketbit & (~str_mask);
}

__attribute__((always_inline)) void QueryProcessor::get_bitmap_rbracket(bitmap& bm) {
    if (bm.has_rbracket == false) {
        build_bitmap_rbracket(bm);
        bm.has_rbracket = true;
    }
}

__attribute__((always_inline)) IntervalInfo QueryProcessor::get_interval(long& pos, unsigned long& bitmap) {
    IntervalInfo itv_info;
    int relative_pos = pos % 64;
    unsigned long w_start = (1UL << relative_pos);
    unsigned long mask_start = w_start ^ (w_start - 1);
    bitmap = bitmap & (~mask_start);
    if (bitmap) {
        unsigned long w_end = bitmap & (-bitmap);
        unsigned long w_interval = (w_end - w_start) | w_end;
        itv_info.intervalbit = w_interval & (~mask_start);
        itv_info.is_complete = true;
    } else {
        // include the last character inside the word (incomplete interval)
        unsigned long w_end = (1UL << 63);
        unsigned long w_interval = (w_end - w_start) | w_end;
        itv_info.intervalbit = w_interval & (~mask_start);
        itv_info.is_complete = false;
    }
    return itv_info;
}

__attribute__((always_inline)) IntervalInfo QueryProcessor::get_interval_new_word(unsigned long& bitmap) {
    IntervalInfo itv_info;
    unsigned long w_start = 1;
    if (bitmap) {
        unsigned long w_end = bitmap & (-bitmap);
        unsigned long w_interval = (w_end - w_start) | w_end;
        itv_info.intervalbit = w_interval;
        itv_info.is_complete = true;
    } else {
        // include the last character inside the word (incomplete interval)
        unsigned long w_end = (1UL << 63);
        unsigned long w_interval = (w_end - w_start) | w_end;
        itv_info.intervalbit = w_interval;
        itv_info.is_complete = false;
    }
    return itv_info;
}

__attribute__((always_inline)) IntervalInfo QueryProcessor::next_interval(unsigned long& bitmap) {
    IntervalInfo itv_info;
    unsigned long w_start = bitmap & (-bitmap);
    bitmap = bitmap & (bitmap - 1);
    if (bitmap) {
        unsigned long w_end = bitmap & (-bitmap);
        unsigned long w_interval = (w_end - w_start) | w_end;
        itv_info.intervalbit = w_interval;
        itv_info.is_complete = true;
    } else {
        // include the last character inside the word (incomplete interval)
        unsigned long w_end = (1UL << 63);
        unsigned long w_interval = (w_end - w_start) | w_end;
        itv_info.intervalbit = w_interval;
        itv_info.is_complete = false;
    }
    return itv_info;
}

__attribute__((always_inline)) long QueryProcessor::get_position(unsigned long& bitmap, int number) {
    while (number > 1) {
        bitmap = bitmap & (bitmap - 1);
        --number;
    }
    unsigned long pos = top_word * 64 + __builtin_ctzll(bitmap);
    return pos;
}

__attribute__((always_inline)) int QueryProcessor::count(unsigned long& interval, unsigned long& bitmap) {
    return __builtin_popcountl(bitmap & interval); 
}

__attribute__((always_inline)) long QueryProcessor::object_end(unsigned long& interval, unsigned long& bitmap) {
    return top_word * 64 + 64 - __builtin_clzll(bitmap & interval);
}

__attribute__((always_inline)) long QueryProcessor::interval_end(unsigned long& interval) {
    return top_word * 64 + 63 - __builtin_clzll(interval);
}

__attribute__((always_inline)) void QueryProcessor::goOverObj(long& pos, bitmap& bm) {
    int num_open = 1;
    int num_close = 0;
    long word_id = pos / 64;
    bool first_interval = false;
    bool new_word = false;
    while (true) {
        while (word_id < mNumWords) {
            if (word_id > top_word) {
                // build basic bitmaps for the next word
                resetBitmap(bm);
                start_id = word_id * 2;
                build_bitmap_basic();
            } 
            get_bitmap_lbrace(bm);
            IntervalInfo interval;
            if (first_interval == false) {
                if (new_word == true) {
                    interval = get_interval_new_word(bm.lbracebit);
                } else {
                    interval = get_interval(pos, bm.lbracebit);
                }
                first_interval = true;
            } else {
                interval = next_interval(bm.lbracebit);
            }
            get_bitmap_rbrace(bm);
            unsigned long bitmap_rbrace = bm.rbracebit & interval.intervalbit;
            num_close = __builtin_popcountl(bitmap_rbrace);
            if (num_close < num_open) {
                if (interval.is_complete == true) {
                    num_open = num_open - num_close + 1;
                    break;
                } else {
                    num_open = num_open - num_close;
                }
            } else {  // results found
                pos = get_position(bitmap_rbrace, num_open);
                return;
            }
            // interval is incomplete in the current word
            ++word_id;
            first_interval = false;
            new_word = true;
        }
    }
}

__attribute__((always_inline)) void QueryProcessor::goOverAry(long& pos, bitmap& bm) {
    int num_open = 1;
    int num_close = 0;
    long word_id = pos / 64;
    bool first_interval = false;
    bool new_word = false;
    while (true) {
        while (word_id < mNumWords) {
            if (word_id > top_word) {
                // build basic bitmaps for the next word
                resetBitmap(bm);
                start_id = word_id * 2;
                build_bitmap_basic();
            }
            get_bitmap_lbracket(bm);
            IntervalInfo interval;
            if (first_interval == false) {
                if (new_word == true) {
                    interval = get_interval_new_word(bm.lbracketbit);
                } else {
                    interval = get_interval(pos, bm.lbracketbit);
                } 
                first_interval = true;
            } else {
                interval = next_interval(bm.lbracketbit);
            } 
            get_bitmap_rbracket(bm);
            unsigned long bitmap_rbracket = bm.rbracketbit & interval.intervalbit;
            bitset<64> tempbit1(bm.rbracketbit);
            num_close = __builtin_popcountl(bitmap_rbracket);
            if (num_close < num_open) {
                if (interval.is_complete == true) {
                    num_open = num_open - num_close + 1;
                    break;
                } else {
                    num_open = num_open - num_close;
                }
            } else {  // results found
                pos = get_position(bitmap_rbracket, num_open);  //bm.rbracebit
                return;
            }
            // interval is incomplete in the current word
            ++word_id;
            first_interval = false;
            new_word = true;
        }
    }
}

__attribute__((always_inline)) void QueryProcessor::goToObjEnd(long& pos, bitmap& bm) {
    int num_open = 1;
    int num_close = 0;
    long word_id = pos / 64;
    bool first_interval = false;
    bool new_word = false;
    while (true) {
        while (word_id < mNumWords) {
            if (word_id > top_word) {
                resetBitmap(bm);
                start_id = word_id * 2;
                build_bitmap_basic();
            } 
            get_bitmap_lbrace(bm);
            IntervalInfo interval;
            if (first_interval == false) {
                if (new_word == true) {
                    interval = get_interval_new_word(bm.lbracebit);
                } else {
                    interval = get_interval(pos, bm.lbracebit);
                }
                first_interval = true;
            } else {
                interval = next_interval(bm.lbracebit);
            }
            get_bitmap_rbrace(bm);
            unsigned long bitmap_rbrace = bm.rbracebit & interval.intervalbit;
            num_close = __builtin_popcountl(bitmap_rbrace);
            if (num_close < num_open) {
                if (interval.is_complete == true) {
                    num_open = num_open - num_close + 1;
                    break;
                } else {
                    num_open = num_open - num_close;
                }
            } else {  // results found
                pos = get_position(bitmap_rbrace, num_open);
                return;
            }
            // interval is incomplete in the current word
            ++word_id;
            first_interval = false;
            new_word = true;
        }
    }
}

__attribute__((always_inline)) void QueryProcessor::goToAryEnd(long& pos, bitmap& bm) {
    int num_open = 1;
    int num_close = 0;
    long word_id = pos / 64;
    bool first_interval = false;
    bool new_word = false;
    while (true) {
        while (word_id < mNumWords) {
            if (word_id > top_word) {
                // build basic bitmaps for the next word
                resetBitmap(bm);
                start_id = word_id * 2;
                build_bitmap_basic();
            }
            get_bitmap_lbracket(bm);
            IntervalInfo interval;
            if (first_interval == false) {
                if (new_word == true) {
                    interval = get_interval_new_word(bm.lbracketbit);
                } else {
                    interval = get_interval(pos, bm.lbracketbit);
                }
                first_interval = true;
            } else {
                interval = next_interval(bm.lbracketbit);
            }
            get_bitmap_rbracket(bm);
            unsigned long bitmap_rbracket = bm.rbracketbit & interval.intervalbit;
            num_close = __builtin_popcountl(bitmap_rbracket);
            if (num_close < num_open) {
                if (interval.is_complete == true) {
                    num_open = num_open - num_close + 1;
                    break;
                } else {
                    num_open = num_open - num_close;
                }
            } else {  // results found
                pos = get_position(bitmap_rbracket, num_open);
                return;
            }
            // interval is incomplete in the current word
            ++word_id;
            first_interval = false;
            new_word = true;
        }
    }
}

__attribute__((always_inline)) void QueryProcessor::goOverPriAttr(long& pos, bitmap& bm) {
    long word_id = pos / 64;
    bool first_interval = false;
    bool new_word = false;
    while (word_id < mNumWords) {
        if (word_id > top_word) {
            // build basic bitmaps for the next word
            resetBitmap(bm);
            start_id = word_id * 2;
            build_bitmap_basic();
        }
        get_bitmap_comma(bm);
        IntervalInfo interval;
        if (first_interval == false) {
            if (new_word == true) {
                interval = get_interval_new_word(bm.commabit);
            } else {
                interval = get_interval(pos, bm.commabit);
            }
            first_interval = true;
        } else {
            interval = next_interval(bm.commabit);
        }
        get_bitmap_rbrace(bm);
        unsigned long bitmap_rbrace = bm.rbracebit & interval.intervalbit;
        if (bitmap_rbrace) {
            // end of object
            pos = get_position(bitmap_rbrace, 1) - 1;
            return;
        }
        if (interval.is_complete) {
            // position before comma
            pos = interval_end(interval.intervalbit) - 1;
            return;
        } 
        // interval is incomplete in the current word
        ++word_id;
        first_interval = false;
        new_word = true;
    }  
}

__attribute__((always_inline)) int QueryProcessor::goOverPriElem(long& pos, bitmap& bm) {
    long word_id = pos / 64;
    bool first_interval = false;
    bool new_word = false;
    while (word_id < mNumWords) {
        if (word_id > top_word) {
            // build basic bitmaps for the next word
            resetBitmap(bm);
            start_id = word_id * 2;
            build_bitmap_basic();
        }
        get_bitmap_comma(bm);
        IntervalInfo interval;
        if (first_interval == false) {
            if (new_word == true) {
                interval = get_interval_new_word(bm.commabit);
            } else {
                interval = get_interval(pos, bm.commabit);
            }
            first_interval = true;
        } else {
            interval = next_interval(bm.commabit);
        }
        get_bitmap_rbracket(bm);
        unsigned long bitmap_rbracket = bm.rbracketbit & interval.intervalbit;
        if (bitmap_rbracket) {
            pos = get_position(bitmap_rbracket, 1);
            return ARRAY_END;
        }
        if (interval.is_complete) {
            // position before comma
            pos = interval_end(interval.intervalbit);
            pos = pos - 1;
            return SUCCESS;
        }
        // interval is incomplete in the current word
        ++word_id;
        first_interval = false;
        new_word = true;
    }
}

__attribute__((always_inline)) int QueryProcessor::goOverPriElems(long& pos, bitmap& bm) {
    long word_id = pos / 64;
    bool new_word = false;
    while (word_id < mNumWords) {
        if (word_id > top_word) {
            // build basic bitmaps for the next word
            resetBitmap(bm);
            start_id = word_id * 2;
            build_bitmap_basic();
        }
        get_bitmap_lbrace(bm);
        get_bitmap_lbracket(bm);
        unsigned long bitmap_bracket = bm.lbracebit | bm.lbracketbit;
        IntervalInfo interval;
        if (new_word == true) {
            interval = get_interval_new_word(bitmap_bracket);
        } else {
            interval = get_interval(pos, bitmap_bracket);
        }
        get_bitmap_rbracket(bm);
        unsigned long bitmap_rbracket = bm.rbracketbit & interval.intervalbit;
        if (bitmap_rbracket) {
            pos = get_position(bitmap_rbracket, 1);
            return ARRAY_END;
        }
        if (interval.is_complete) {
            pos = interval_end(interval.intervalbit);
            return SUCCESS;
        }
        ++word_id;
        new_word = true;
    }
}

__attribute__((always_inline)) int QueryProcessor::goToObjElem(long& pos, bitmap& bm) {
    do {
        if (mRecord[pos] != '{' || mRecord[pos] != '[') {
        int result = goOverPriElems(pos, bm);
        if (result == ARRAY_END) {
            return result;
        }
        }
        int element_type = getElementType(pos);
        if (element_type == OBJECT) {
            return SUCCESS;
        }
        goOverAry(pos, bm);
    } while (hasMoreElements(pos));
    return OBJECT_END;
}

__attribute__((always_inline)) int QueryProcessor::goToAryElem(long& pos, bitmap& bm) {
    do {
        if (mRecord[pos] != '{' || mRecord[pos] != '[') {
            int result = goOverPriElems(pos, bm);
            if (result == ARRAY_END) {
                return result;
            }
        }
        int element_type = getElementType(pos);
        if (element_type == ARRAY) {
            return SUCCESS;
        }
        goOverObj(pos, bm);
    } while (hasMoreElements(pos));
    return OBJECT_END;
}

__attribute__((always_inline)) int QueryProcessor::goOverPriAttrs(long& pos, bitmap& bm) {
    long word_id = pos / 64;
    bool new_word = false;
    while (word_id < mNumWords) {
        if (word_id > top_word) {
            // build basic bitmaps for the next word
            resetBitmap(bm);
            start_id = word_id * 2;
            build_bitmap_basic();
        }
        get_bitmap_lbrace(bm);
        get_bitmap_lbracket(bm);
        unsigned long bitmap_bracket = bm.lbracebit | bm.lbracketbit; 
        IntervalInfo interval;
        if (new_word == true) {
            interval = get_interval_new_word(bitmap_bracket);
        } else {
            interval = get_interval(pos, bitmap_bracket);
        }
        get_bitmap_rbrace(bm);
        unsigned long bitmap_rbrace = bm.rbracebit & interval.intervalbit;
        if (bitmap_rbrace) {
            pos = get_position(bitmap_rbrace, 1);
            return OBJECT_END;
        }
        if (interval.is_complete) {
            pos = interval_end(interval.intervalbit);
            return SUCCESS;
        }
        ++word_id;
        new_word = true; 
    }
}

__attribute__((always_inline)) int QueryProcessor::goToObjAttr(long& pos, bitmap& bm) {
    do {
        int result = goOverPriAttrs(pos, bm);
        if (result == OBJECT_END) {
            return result;
        }
        int attribute_type = getAttributeType(pos);
        if (attribute_type == OBJECT) {
            return SUCCESS;
        }
        goOverAry(pos, bm);
    } while (hasMoreAttributes(pos));
    return OBJECT_END;
}

__attribute__((always_inline)) int QueryProcessor::goToAryAttr(long& pos, bitmap& bm) {
    do {
        int result = goOverPriAttrs(pos, bm);
        if (result == OBJECT_END) {
            return result;
        }
        int attribute_type = getAttributeType(pos);
        if (attribute_type == ARRAY) {
            return SUCCESS;
        }
        goOverObj(pos, bm);
    } while (hasMoreAttributes(pos));
    return OBJECT_END;
}

__attribute__((always_inline)) int QueryProcessor::goToPrimAttr(long& pos, bitmap& bm) {
    long word_id = pos / 64;
    bool first_interval = false;
    bool new_word = false;
    while (true) {
        while (word_id < mNumWords) {
            if (word_id > top_word) {
                // build basic bitmaps for the next word
                resetBitmap(bm);
                start_id = word_id * 2;
                build_bitmap_basic();
            }
            get_bitmap_colon(bm);
            IntervalInfo interval;
            if (first_interval == false) {
                if (new_word == true) {
                    interval = get_interval_new_word(bm.colonbit);
                    new_word = false;
                } else {
                    interval = get_interval(pos, bm.colonbit);
                }
                first_interval = true;
            } else {
                interval = next_interval(bm.colonbit);
            }
            get_bitmap_rbrace(bm);
            unsigned long bitmap_rbrace = bm.rbracebit & interval.intervalbit;
            if (bitmap_rbrace > 0) {
                // object ends
                pos = get_position(bitmap_rbrace, 1);
                return OBJECT_END;
            }
            if (interval.is_complete) {
                pos = interval_end(interval.intervalbit) + 1;
                int type = getAttributeType(pos);
                if (type == OBJECT) {
                    goOverObj(pos, bm);
                    word_id = pos / 64;  // update word id 
                    first_interval = false;
                }
                else if (type == ARRAY) {
                    goOverAry(pos, bm);
                    word_id = pos / 64; // update word id 
                    first_interval = false;
                }
                else {
                    return SUCCESS;
                }
                break;
            }
            ++word_id;
            first_interval = false;
            new_word = true;
        }
    }
}

__attribute__((always_inline)) JumpInfo QueryProcessor::goOverPrimElemsInRange(long& pos, int num_elements, bitmap& bm) {
    int word_id = pos / 64;
    bool new_word = false;
    int num_comma = 0;
    while (word_id < mNumWords) {
        if (word_id > top_word) {
            // build basic bitmaps for the next word
            resetBitmap(bm);
            start_id = word_id * 2;
            build_bitmap_basic();
        }
        get_bitmap_lbrace(bm);
        get_bitmap_lbracket(bm);
        unsigned long bitmap_bracket = bm.lbracebit | bm.lbracketbit;
        IntervalInfo interval;
        if (new_word == true) {
            interval = get_interval_new_word(bitmap_bracket);
        } else {
            interval = get_interval(pos, bitmap_bracket);
        }
        build_bitmap_rbracket(bm);
        unsigned long bitmap_rbracket = bm.rbracketbit & interval.intervalbit;
        get_bitmap_comma(bm);
        unsigned long bitmap_comma = bm.commabit & interval.intervalbit;
        if (bitmap_rbracket) {
            bitmap_comma = bitmap_comma & (bitmap_rbracket ^ (bitmap_rbracket - 1));
        }
        num_comma = num_comma +__builtin_popcountl(bitmap_comma);
        if (num_comma >= num_elements) {
            long temp_pos = word_id * 64 + __builtin_ctzll(bitmap_comma);
            pos = get_position(bitmap_comma, num_elements);
            JumpInfo ji(SUCCESS);
            return ji;
        }
        if (bitmap_rbracket) {
            // end of array
            pos = get_position(bitmap_rbracket, 1);
            JumpInfo ji(ARRAY_END);
            return ji;
        } else {
            if (interval.is_complete) {
                pos = interval_end(interval.intervalbit);// + 1;
                JumpInfo ji(PARTIAL_SKIP, num_comma);
                return ji;
            }
            num_elements -= num_comma;
        }
        // interval is incomplete in the current word
        ++word_id;
        new_word = true;
    }
}


__attribute__((always_inline)) int QueryProcessor::goOverElem(long& pos, int num_elements, bitmap& bm) {
    while (num_elements > 0) {
        if (!hasMoreElements(pos)) {
            return ARRAY_END;
        }
        int element_type = getElementType(pos);
        int result = 0;
        switch(element_type) {
            case PRIMITIVE: {
                JumpInfo res = goOverPrimElemsInRange(pos, num_elements, bm);
                if (res.status == ARRAY_END || res.status == SUCCESS) {
                    return res.status;
                }
                if (res.status == PARTIAL_SKIP) {
                    num_elements = num_elements - res.num_comma + 1;
                }
                break;
            }
            case OBJECT:
                goOverObj(pos, bm);
                break;
            case ARRAY:
                goOverAry(pos, bm);
                break;
        }
        --num_elements;
    }
    return SUCCESS;
}

__attribute__((always_inline)) int QueryProcessor::goToObjElemInRange(long& pos, int& num_elements, bitmap& bm) {
     do {
        int element_type = getElementType(pos);
        int result = 0;
        switch(element_type) {
            case PRIMITIVE: {
                JumpInfo res = goOverPrimElemsInRange(pos, num_elements, bm);
                if (res.status == ARRAY_END) {
                    return res.status;
                }
                if (res.status == SUCCESS) {
                    return RANGE_END;
                }
                if (res.status == PARTIAL_SKIP) {
                    num_elements = num_elements - res.num_comma + 1;
                }
                break;
            }
            case OBJECT:
                return SUCCESS;
            case ARRAY:
                goOverAry(pos, bm);
                break;
        }
        --num_elements;
        if (!hasMoreElements(pos)) {
            return ARRAY_END;
        }
    } while (num_elements > 0);
    return RANGE_END;
}

__attribute__((always_inline)) int QueryProcessor::goToAryElemInRange(long& pos, int& num_elements, bitmap& bm) {
    do {
        int element_type = getElementType(pos);
        int result = 0;
        switch(element_type) {
            case PRIMITIVE: {
                JumpInfo res = goOverPrimElemsInRange(pos, num_elements, bm);
                if (res.status == ARRAY_END) {
                    return res.status;
                }
                if (res.status == SUCCESS) {
                    return RANGE_END;
                }
                if (res.status == PARTIAL_SKIP) {
                    num_elements = num_elements - res.num_comma + 1;
                }
                break;
            }
            case OBJECT:
                goOverObj(pos, bm);
                break;
            case ARRAY:
                return SUCCESS;
        }
        --num_elements;
        if (!hasMoreElements(pos)) {
            return ARRAY_END;
        }
    } while (num_elements > 0);
    return RANGE_END;
}

__attribute__((always_inline)) int QueryProcessor::goToPrimElemInRange(long& pos, int& num_elements, bitmap& bm) {
    do {
        int element_type = getElementType(pos);
        int result = 0;
        switch(element_type) {
            case PRIMITIVE: {
                return SUCCESS;
            }
            case OBJECT:
                goOverObj(pos, bm);
                break;
            case ARRAY:
                goOverAry(pos, bm);
        }
        --num_elements;
        if (!hasMoreElements(pos)) {
            return ARRAY_END;
        }
    } while (num_elements > 0);
    return RANGE_END;
}

__attribute__((always_inline)) bool QueryProcessor::hasMoreElements(long& pos) {
    while (mRecord[pos] == ' ' || mRecord[pos] == '\n' || mRecord[pos] == '\r') ++pos;
    ++pos;
    while (mRecord[pos] == ' ' || mRecord[pos] == '\n' || mRecord[pos] == '\r') ++pos; 
    if (mRecord[pos] == ']') {
        return false;
    }
    if (mRecord[pos] == ',') ++pos;
    while (mRecord[pos] == ' ' || mRecord[pos] == '\n' || mRecord[pos] == '\r') ++pos;
    return true;
}

__attribute__((always_inline)) int QueryProcessor::getElementType(long& pos) {
    while (mRecord[pos] == ' ') ++pos;
    if (mRecord[pos] == '{') return OBJECT;
    if (mRecord[pos] == '[') return ARRAY;
    return PRIMITIVE;
}

__attribute__((always_inline)) int QueryProcessor::goToPrimElem(long& pos, bitmap& bm) {
    do {
        int element_type = getElementType(pos);
        switch (element_type) {
            case PRIMITIVE:
                return SUCCESS;
            case OBJECT:
                goOverObj(pos, bm);
                break;
            case ARRAY:
                goOverAry(pos, bm);
        }
    } while (hasMoreElements(pos));
    return ARRAY_END;
}

__attribute__((always_inline)) bool QueryProcessor::hasMoreAttributes(long& pos) {
    // if current character is blank, skip this character until meeting a non-blank character
    while (mRecord[pos] == ' ') ++pos;
    ++pos;
    while (mRecord[pos] == ' ') {
        ++pos;
    }
    if (mRecord[pos] == '}') {
        return false;
    }
    if (mRecord[pos] == ',') ++pos;
    while (mRecord[pos] == ' ' || mRecord[pos] == '\n') ++pos; 
    return true;
}

__attribute__((always_inline)) int QueryProcessor::getAttributeType(long& pos) {
    while (mRecord[pos] == ' ') ++pos;
    if (mRecord[pos] == '{') return OBJECT;
    if (mRecord[pos] == '[') return ARRAY;
    return PRIMITIVE;
}

void QueryProcessor::object(long& pos, bitmap& bm) {
    int attribute_type = qa.typeExpectedInObj();
    while (hasMoreAttributes(pos)) {
        int result = 0;
        int next_state = 0;
        int element_type = attribute_type;
        switch (attribute_type) {
            case OBJECT:
                result = goToObjAttr(pos, bm);
                break;
            case ARRAY:
                result = goToAryAttr(pos, bm);
                break;
            case PRIMITIVE: {
                long st = pos;
                while (mRecord[st] != '"') ++st;
                long ed = st + 1;
                while (mRecord[ed] != '"') ++ed;
                int key_len = ed - st - 1;
                memcpy(mKey, mRecord + st + 1, key_len);
                mKey[key_len] = '\0';
                next_state = qa.getNextState(mRecord + st + 1, key_len);
                while (mRecord[ed] != ':') ++ed; 
                pos = ed + 1;
                element_type = getElementType(pos);
            }
        }
        if (result == OBJECT_END)
            return;
        if (attribute_type != PRIMITIVE) {
            long st = pos;
            while (mRecord[st] != ':') --st;
            while (mRecord[st] != '"') --st;
            long ed = st - 1;
            while (mRecord[ed] != '"') --ed;
            int key_len = st - ed - 1;
            memcpy(mKey, mRecord + ed + 1, key_len);
            mKey[key_len] = '\0';
            next_state = qa.getNextState(mRecord + ed + 1, key_len);
        }
        if (next_state == UNMATCHED_STATE) {
            switch (element_type) {
                case OBJECT:
                    goOverObj(pos, bm);
                    break;
                case ARRAY:
                    goOverAry(pos, bm);
                    break;
                case PRIMITIVE: {
                    goOverPriAttr(pos, bm);
                }
            }
        } else if (qa.isAccept(next_state) == true) { //ACCEPT
            ++mNumMatches;
            long start_pos = pos;
            switch (element_type) {
                case OBJECT: {
                    goOverObj(pos, bm);
                    break;
                }
                case ARRAY: {
                    goOverAry(pos, bm);
                    break;
                }
                case PRIMITIVE:
                    goOverPriAttr(pos, bm);
                    ++pos;
            }
            if (mRecord[pos] != '}') {
                if (qa.getStackSize() == 0) return;
                goToObjEnd(pos, bm);
            }
            break;
        } else {  // in-progress
            qa.pushStack(next_state);
            switch (attribute_type) {
                case OBJECT:
                    object(pos, bm);
                    break;
                case ARRAY:
                    array(pos, bm);
            }
            qa.popStack(); // virtual token "value"
            if (qa.getStackSize() == 0) return;
            goToObjEnd(pos, bm);
            break;
        }
    }
}

void QueryProcessor::array(long& pos, bitmap& bm) {
    int next_state = qa.getNextStateNoKey();
    qa.pushStack(next_state);
    int element_type = qa.typeExpectedInArr();
    long prev_pos = -1; // only use for debugging
    if (qa.hasIndexConstraints()) {
        IndexInfo idx_info = qa.getIndexInfo(qa.mCurState);
        int start_idx = idx_info.start_idx;
        int end_idx = idx_info.end_idx;
        int num_elements = end_idx - start_idx;
        if (start_idx > 0) {
            int result = goOverElem(pos, start_idx, bm);
            if (result == ARRAY_END) {
                qa.popStack();
                return; 
            }
        }
        while (hasMoreElements(pos) && num_elements > 0) {
            if (qa.isAccept(qa.mCurState) == true) {
                ++mNumMatches;
                long start_pos = pos;
                bool break_while = false;
                int value_type = element_type;
                if (element_type == PRIMITIVE) {
                    value_type = getElementType(pos); 
                }
                switch (value_type) {
                    case OBJECT: {
                        goOverObj(pos, bm);
                        break;
                    }
                    case ARRAY: {
                        goOverAry(pos, bm);
                        break;
                    }
                    case PRIMITIVE: {
                        int result = goOverPriElem(pos, bm);
                        if (result == ARRAY_END) {
                            break_while = true;
                        }
                    }
                }
                if (break_while) {
                    if (mRecord[pos] != ']')
                        goToAryEnd(pos, bm);
                    break;
                }
                --num_elements;
            } else if (qa.mCurState > 0) {
                int result; 
                switch (element_type) {
                    case OBJECT: {
                        result = goToObjElemInRange(pos, num_elements, bm);
                        break;
                    }
                    case ARRAY: {
                        result = goToAryElemInRange(pos, num_elements, bm);
                    }
                }
                if (result == SUCCESS) {
                    switch (element_type) {
                        case OBJECT:
                            prev_pos = pos;
                            object(pos, bm);
                            break;
                        case ARRAY: {
                            array(pos, bm);
                        }
                    }
                    --num_elements;
                } else if (result == ARRAY_END) {
                    qa.popStack();
                    return;
                } else if (result == RANGE_END) {
                    if (mRecord[pos] != ']') {
                        if (qa.getStackSize() == 1) return;
                        goToAryEnd(pos, bm);
                    }
                    break;
                }
            }
        }
        if (mRecord[pos] != ']') {
            if (qa.getStackSize() == 1) return;
            goToAryEnd(pos, bm);
        }
    } else {
        while (hasMoreElements(pos)) {
            if (qa.isAccept(qa.mCurState) == true) {
                ++mNumMatches;
                long start_pos = pos;
                bool break_while = false;
                int value_type = element_type;
                if (element_type == PRIMITIVE)
                    value_type = getElementType(pos);
                switch (value_type) {
                    case OBJECT: {
                        goOverObj(pos, bm);
                        break;
                    }
                    case ARRAY: {
                        goOverAry(pos, bm);
                        break;
                    }
                    case PRIMITIVE: {
                        int result = goOverPriElem(pos, bm);
                        if (result == ARRAY_END) {
                            break_while = true;
                        }
                    }
                }
                if (break_while) break;
            } else if (qa.mCurState > 0) {
                if (getElementType(pos) != element_type) {
                    int result;
                    switch (element_type) {
                        case OBJECT:
                            result = goToObjElem(pos, bm);
                            break;
                        case ARRAY:
                            result = goToAryElem(pos, bm);
                    }
                    if (result == ARRAY_END) {
                        qa.popStack();
                        return;
                    }
                }
                switch (element_type) {
                    case OBJECT:
                        prev_pos = pos;
                        object(pos, bm);
                        break;
                    case ARRAY: {
                        array(pos, bm);;
                    }
                }
            }
        }
    }
    qa.popStack();
}

char QueryProcessor::getNextNonEmptyCharacter(long& pos) {
    char ch = mRecord[pos];
    while (mRecord[pos] == ' ') ++pos;
    return mRecord[pos];
}

long QueryProcessor::getOutputMatchesNum() {
    return mNumMatches;
}

long QueryProcessor::runQuery(Record* rec) {
    setRecordText(rec->text + rec->rec_start_pos, rec->rec_length);
    init();
    long cur_pos = 0;
    char ch = getNextNonEmptyCharacter(cur_pos);
    bitmap bm;
    if (ch == '{' && qa.typeExpectedInObj() != NONE)
        object(cur_pos, bm);
    else if(ch == '[' && qa.typeExpectedInArr() != NONE)
        array(cur_pos, bm);
    return mNumMatches;
}
