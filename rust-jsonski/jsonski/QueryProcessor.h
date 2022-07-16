#ifndef QUERYPROCESSOR_H
#define QUERYPROCESSOR_H
#include <string>
#include <iostream>
#include <vector>
#include <bitset>
#include <cassert>
#include <stack>
#include <algorithm>
#include <unordered_map>
#include <functional>
#include <math.h>
#include <immintrin.h>
#include <unordered_map>
#include <map>
#include "JSONPathParser.h"
#include "QueryAutomaton.h"
#include "Records.h"
using namespace std;

#define SUCCESS 1001
#define ARRAY_END 1002
#define OBJECT_END 1003
#define RANGE_END 1004
#define PARTIAL_SKIP 1005
#define MAX_KEY_LENGTH 1000
#define MAX_TEXT_LENGTH 10000


#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

typedef struct bitmap{
    unsigned long colonbit = 0;
    unsigned long commabit = 0;
    unsigned long lbracebit = 0;
    unsigned long rbracebit = 0;
    unsigned long lbracketbit = 0;
    unsigned long rbracketbit = 0;
    bool has_colon = false;
    bool has_comma = false;
    bool has_lbrace = false;
    bool has_rbrace = false;
    bool has_lbracket = false;
    bool has_rbracket = false;
} bitmap;

typedef struct IntervalInfo {
    unsigned long intervalbit = 0;
    bool is_complete = true;
} IntervalInfo;

struct JumpInfo {
    int status;
    int num_comma;
    JumpInfo(int s, int n = 0) {
        status = s;
        num_comma = n;
    }
};

class QueryProcessor {
  public:
    // initialization. including query automaton construction and
    // some internal variables initialization for supporting bit-parallel
    // fast-forwarding optimizations.
    QueryProcessor(string query);
    ~QueryProcessor();
    long getOutputMatchesNum();
    // execute query on one single JSON record
    long runQuery(Record* rec);

  private:
    void init();
    void setRecordText(char* rec_text, long record_length);
    char getNextNonEmptyCharacter(long& pos);
    void object(long& pos, bitmap& bm);
    void array(long& pos, bitmap& bm);
    void array_range(long& pos, bitmap& bm);
    
    // fast-forward functions
    __attribute__((always_inline)) void goOverObj(long& pos, bitmap& bm);
    __attribute__((always_inline)) void goOverAry(long& pos, bitmap& bm);
    __attribute__((always_inline)) void goOverPriAttr(long& pos, bitmap& bm);
    __attribute__((always_inline)) int goOverPriElem(long& pos, bitmap& bm); 
    __attribute__((always_inline)) void goToObjEnd(long& pos, bitmap& bm);
    __attribute__((always_inline)) void goToAryEnd(long& pos, bitmap& bm);
    __attribute__((always_inline)) int goOverElem(long& pos, int num_elements, bitmap& bm);
    __attribute__((always_inline)) JumpInfo goOverPrimElemsInRange(long& pos, int num_elements, bitmap& bm);
    __attribute__((always_inline)) int goToObjElemInRange(long& pos, int& num_elements, bitmap& bm);
    __attribute__((always_inline)) int goToAryElemInRange(long& pos, int& num_elements, bitmap& bm);
    __attribute__((always_inline)) int goToPrimElemInRange(long& pos, int& num_elements, bitmap& bm);
    __attribute__((always_inline)) int goToObjAttr(long& pos, bitmap& bm);
    __attribute__((always_inline)) int goToAryAttr(long& pos, bitmap& bm);
    __attribute__((always_inline)) int goToPrimAttr(long& pos, bitmap& bm);
    __attribute__((always_inline)) int goToObjElem(long& pos, bitmap& bm);
    __attribute__((always_inline)) int goToAryElem(long& pos, bitmap& bm);
    __attribute__((always_inline)) int goToPrimElem(long& pos, bitmap& bm);
    __attribute__((always_inline)) int goOverPriAttrs(long& pos, bitmap& bm);
    __attribute__((always_inline)) int goOverPriElems(long& pos, bitmap& bm); 
    __attribute__((always_inline)) bool hasMoreElements(long& pos);
    __attribute__((always_inline)) int getElementType(long& pos);
    __attribute__((always_inline)) bool hasMoreAttributes(long& pos);
    __attribute__((always_inline)) int getAttributeType(long& pos);

    // structural interval construction and access 
    __attribute__((always_inline)) void resetBitmap(bitmap& bm) {
        bm.has_colon = false;
        bm.has_comma = false;
        bm.has_lbrace = false;
        bm.has_rbrace = false;
        bm.has_lbracket = false;
        bm.has_rbracket = false;
    }    
    // first three steps of structral index construction, get string mask bitmap
    __attribute__((always_inline)) void build_bitmap_basic(); 
    __attribute__((always_inline)) void build_bitmap_colon(bitmap& bm);
    __attribute__((always_inline)) void build_bitmap_comma(bitmap& bm);
    __attribute__((always_inline)) void build_bitmap_lbrace(bitmap& bm);
    __attribute__((always_inline)) void build_bitmap_rbrace(bitmap& bm);
    __attribute__((always_inline)) void build_bitmap_lbracket(bitmap& bm);
    __attribute__((always_inline)) void build_bitmap_rbracket(bitmap& bm);
    __attribute__((always_inline)) void get_bitmap_colon(bitmap& bm);
    __attribute__((always_inline)) void get_bitmap_comma(bitmap& bm);
    __attribute__((always_inline)) void get_bitmap_lbrace(bitmap& bm);
    __attribute__((always_inline)) void get_bitmap_rbrace(bitmap& bm);
    __attribute__((always_inline)) void get_bitmap_lbracket(bitmap& bm);
    __attribute__((always_inline)) void get_bitmap_rbracket(bitmap& bm);
    __attribute__((always_inline)) IntervalInfo get_interval_new_word(unsigned long& bitmap);
    __attribute__((always_inline)) IntervalInfo get_interval(long& pos, unsigned long& bitmap);
    __attribute__((always_inline)) IntervalInfo next_interval(unsigned long& bitmap);
    __attribute__((always_inline)) long get_position(unsigned long& bitmap, int number); 
    __attribute__((always_inline)) long interval_end(unsigned long& interval);
    __attribute__((always_inline)) void get_interval_brace(long& pos, bitmap& bm, IntervalInfo& itv_info);
    __attribute__((always_inline)) void next_interval_brace(bitmap& bm, IntervalInfo& itv_info);
    __attribute__((always_inline)) void get_interval_bracket(long& pos, bitmap& bm, IntervalInfo& itv_info);
    __attribute__((always_inline)) void next_interval_bracket(bitmap& bm, IntervalInfo& itv_info); 
    __attribute__((always_inline)) long get_position_brace(bitmap& bm, int number);
    __attribute__((always_inline)) long get_position_bracket(bitmap& bm, int number);
    __attribute__((always_inline)) void next_interval(char ch);
    __attribute__((always_inline)) int count(unsigned long& interval, unsigned long& bitmap);
    __attribute__((always_inline)) long object_end(unsigned long& interval, unsigned long& bitmap);

    // all private variables
    unsigned long str_mask;
    unsigned long escapebit, stringbit, lbracebit, rbracebit, lbracketbit, rbracketbit;
    unsigned long bracketbit0, colonbit0, quotebit0, escapebit0, stringbit0, lbracebit0, rbracebit0, commabit0, lbracketbit0, rbracketbit0;
    long start_id;
    __m256i v_text0, v_text;
    int64_t quote_bits; unsigned long st_quotebit; unsigned long ed_quotebit; unsigned long cb_bit;
    __m256i struct_mask;
    __m256i structural_table, v_quote, v_colon, v_escape, v_lbrace, v_rbrace, v_comma, v_lbracket, v_rbracket;
    uint64_t prev_iter_ends_odd_backslash;
    uint64_t prev_iter_inside_quote;
    uint64_t even_bits;
    uint64_t odd_bits;
    unsigned long first, second;
    long top_word;
    unsigned long cb_mask;
    unsigned long colonbit;
    unsigned long quotebit;
    unsigned long commabit;
    unsigned long bracketbit;
    bool cur_word;
    long cur_pos;
    unsigned long mask;
    unsigned long colon_mask;
    unsigned long comma_mask;

    char* mRecord;
    // for a single large record, stream length equals to record length
    long mRecordLength;
    // each temp word has 32 bytes
    long mNumTmpWords;
    // each word has 64 bytes
    long mNumWords;
    char mKey[MAX_KEY_LENGTH];
    char* mText;
    long mWordId;
    QueryAutomaton qa;
    long mNumMatches;
};
#endif
