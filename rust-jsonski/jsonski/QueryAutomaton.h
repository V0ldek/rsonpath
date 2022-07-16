#ifndef QUERYAUTOMATON_H
#define QUERYAUTOMATON_H
#include <iostream>
#include <string.h>
#include <bitset>
using namespace std;

#define MAX_STATES 50
#define MAX_STACK_DEPTH 50
#define MAX_TRANS_STRING 10
#define MAX_KEY_LENGTH 100

#define UNMATCHED_STATE 0
#define START_STATE 1

#define OBJECT 101
#define ARRAY 102
#define PRIMITIVE 103
#define KEY 104
#define ANY 105
#define OUTPUT_CANDIDATE 106
#define GENERAL_OUTPUT 107
#define NONE 108
#define INVALID -1

typedef struct TransStrInfo {
    char key[MAX_KEY_LENGTH];
    int key_len;
    int exp_type_in_obj = NONE;
    int exp_type_in_arr = NONE;
    int next_state;
} TransStrInfo;

typedef struct StateTransEle {
    TransStrInfo t_str_info[MAX_TRANS_STRING];
    int num_trans_str = 1;
    bool matched_state;
    int start_idx;
    int end_idx;
    bool has_index_constraint = false;
} StateTransEle;

typedef struct IndexInfo {
    int start_idx;
    int end_idx;
} IndexInfo;
 
typedef struct DFA {
    StateTransEle trans_ele[MAX_STATES];
} DFA;

typedef struct Stack {
    int stack[MAX_STACK_DEPTH];
    int arr_counter_stack[MAX_STACK_DEPTH];
    int num_stack_ele;
} Stack;

class QueryAutomaton {
  public:
    QueryAutomaton() {
        reset();
    }

    void reset() {
        mStack.num_stack_ele = 0;
        mCurState = 1;
        mArrCounter = -1;
    }

    void updateStateTransInfo(int cur_state, bool is_final_state, int exp_type_in_obj, int exp_type_in_arr, char* exp_key, int next_state) {
        int cur_idx = cur_state - 1;
        int next_idx = next_state - 1;
        int trans_idx = 0;
        if (exp_key != NULL) {
            strcpy(mDfa.trans_ele[cur_idx].t_str_info[trans_idx].key, exp_key);
            mDfa.trans_ele[cur_idx].t_str_info[trans_idx].key_len = strlen(exp_key);
        }
        if (exp_type_in_obj != NONE)
            mDfa.trans_ele[cur_idx].t_str_info[trans_idx].exp_type_in_obj = exp_type_in_obj;
        if (exp_type_in_arr != NONE)
            mDfa.trans_ele[cur_idx].t_str_info[trans_idx].exp_type_in_arr = exp_type_in_arr;
        mDfa.trans_ele[cur_idx].t_str_info[trans_idx].next_state = next_state;
        mDfa.trans_ele[cur_idx].matched_state = is_final_state; 
    }

    void addIndexConstraints(int state, int start_idx, int end_idx) {
        if (state != UNMATCHED_STATE) {
            mDfa.trans_ele[state - 1].has_index_constraint = true;
            mDfa.trans_ele[state - 1].start_idx = start_idx;
            mDfa.trans_ele[state - 1].end_idx = end_idx;
        }
    }

    bool hasIndexConstraints() {
        if (mCurState == UNMATCHED_STATE) return false;
        return mDfa.trans_ele[mCurState - 1].has_index_constraint;
    }

    void addArrayCounter() {
        ++mArrCounter;
    }

    bool checkArrayCounter() {
        if (mCurState == UNMATCHED_STATE) return false;
        int start_idx = mDfa.trans_ele[mCurState - 1].start_idx;
        int end_idx = mDfa.trans_ele[mCurState - 1].end_idx;
        if (mArrCounter >= start_idx && mArrCounter < end_idx) {
            return true;
        }
        return false;
    }

    __attribute__((always_inline)) int typeExpectedInObj() {
        if (mCurState == UNMATCHED_STATE) return false;
        int cur_idx = mCurState - 1;
        return mDfa.trans_ele[cur_idx].t_str_info[0].exp_type_in_obj;
    }

    __attribute__((always_inline)) int typeExpectedInArr() {
        if (mCurState == UNMATCHED_STATE) return false;
        int cur_idx = mCurState - 1;
        return mDfa.trans_ele[cur_idx].t_str_info[0].exp_type_in_arr;
    } 

    IndexInfo getIndexInfo(int state) {
        IndexInfo idx_info;
        if (state == UNMATCHED_STATE) {
            idx_info.start_idx = -1;
            return idx_info;
        }
        idx_info.start_idx = mDfa.trans_ele[state - 1].start_idx;
        idx_info.end_idx = mDfa.trans_ele[state - 1].end_idx;    
        return idx_info;
    }

    __attribute__((always_inline)) int getNextState(char *key, int key_len) {
        if (mCurState == UNMATCHED_STATE) return UNMATCHED_STATE;
        int cur_idx = mCurState - 1;
        int num_trans_str = mDfa.trans_ele[cur_idx].num_trans_str;
        int i = 0;
        int next_state = UNMATCHED_STATE;
        while (i < num_trans_str) {
            if (mDfa.trans_ele[cur_idx].t_str_info[i].key_len == key_len 
                && memcmp(mDfa.trans_ele[cur_idx].t_str_info[i].key, key, key_len) == 0) {
                next_state = mDfa.trans_ele[cur_idx].t_str_info[i].next_state;
                return next_state;
            }
            ++i;
        }
        return next_state;
    }

    __attribute__((always_inline)) int getNextStateNoKey() {
        if (mCurState == UNMATCHED_STATE) return UNMATCHED_STATE;
        int cur_idx = mCurState - 1;
        int num_trans_str = mDfa.trans_ele[cur_idx].num_trans_str;
        int i = 0;
        int next_state = UNMATCHED_STATE;
        while (i < num_trans_str) {
            if (mDfa.trans_ele[cur_idx].t_str_info[i].key_len == 0) {
                next_state = mDfa.trans_ele[cur_idx].t_str_info[i].next_state;
                return next_state;
            }
            ++i;
        }
        return next_state;
    }

    void setCurState(int cur_state) {
        mCurState = cur_state;
    }

    int getType(int state) {
        if (state != UNMATCHED_STATE && mDfa.trans_ele[state - 1].matched_state == true)
            return OUTPUT_CANDIDATE;
        return GENERAL_OUTPUT;
    }

    int isAccept(int state) {
        if (state != UNMATCHED_STATE && mDfa.trans_ele[state - 1].matched_state == true)
            return true;
        return false;
    }

    __attribute__((always_inline)) void pushStack(int next_state) {
        if (mStack.num_stack_ele < MAX_STACK_DEPTH) {
            mStack.stack[mStack.num_stack_ele] = mCurState;
            mStack.arr_counter_stack[mStack.num_stack_ele++] = mArrCounter;
            mCurState = next_state;
            mArrCounter = -1;
        } else {
            cout<<"exception: stack is empty "<<endl;
        }
    }

    __attribute__((always_inline)) int popStack() {
        if (mStack.num_stack_ele > 0) {
            mCurState = mStack.stack[--mStack.num_stack_ele];
            mArrCounter = mStack.arr_counter_stack[mStack.num_stack_ele];
            return mCurState;
        }
        cout<<"pop out exception "<<endl;
        return INVALID;
    }

    int getStackSize() {
        return mStack.num_stack_ele;
    }

    ~QueryAutomaton() {} 
   
  public:
    int mCurState;

  private:
    DFA mDfa;
    int mArrCounter = -1;
    Stack mStack;
};
#endif
