#include "JSONPathParser.h"
#include <stdlib.h>
#include <limits.h>

void JSONPathParser::updateQueryAutomaton(string query, QueryAutomaton &qa) {
    int length = query.size();
    int lexer_state = 0;
    int query_state = START_STATE;
    char buffer[MAX_KEY_LENGTH];
    for (int i = 0; i < length; ++i) {
        char ch = query[i];
        switch (lexer_state) {
            case 0: { // begin of the path
                if (ch == '.') {
                    lexer_state = 1;
                } else if (ch == '[') {
                    lexer_state = 2;
                    qa.updateStateTransInfo(query_state, false, NONE, ARRAY, NULL, query_state + 1);
                    // cout<<"("<<query_state<<", false, NONE, ARRAY, NULL, "<<(query_state + 1)<<")"<<endl;
                    ++query_state;
                }
                break;
            }
            case 1: { // in object
                int key_end = 0;
                while (ch != '.' && ch != '[') {
                    buffer[key_end++] = ch;
                    if (i + 1 == length) break;
                    ch = query[++i];
                }
                buffer[key_end] = '\0';
                if (i + 1 < length) {
                    if (ch == '[') {
                        lexer_state = 2;
                        // current query state -> expected key-array pair
                        qa.updateStateTransInfo(query_state, false, ARRAY, NONE, buffer, query_state + 1);
                        // cout<<"("<<query_state<<", false, ARRAY, NONE, "<<buffer<<", "<<(query_state + 1)<<")"<<endl;
                        // state transition for [
                        qa.updateStateTransInfo(query_state + 1, false, NONE, NONE, NULL, query_state + 2);
                        // cout<<"("<<(query_state + 1)<<", false, NONE, NONE, NULL, "<<(query_state + 2)<<")"<<endl;
                        query_state += 2;
                        // break;
                    } else if (ch == '.') {
                        lexer_state = 1;
                        qa.updateStateTransInfo(query_state, false, OBJECT, NONE, buffer, query_state + 1);
                        // cout<<"("<<query_state<<", false, OBJECT, NONE, "<<buffer<<", "<<(query_state + 1)<<")"<<endl;
                        ++query_state;
                        // break;
                    } 
                } else {
                    // output info
                    qa.updateStateTransInfo(query_state, false, PRIMITIVE, NONE, buffer, query_state + 1);
                    // cout<<"("<<query_state<<", false, PRIMITIVE, NONE, "<<buffer<<", "<<(query_state + 1)<<")"<<endl;
                    qa.updateStateTransInfo(query_state + 1, true, NONE, NONE, NULL, query_state + 1);
                    // cout<<"("<<(query_state + 1)<<", true, NONE, NONE, NULL, "<<(query_state + 1)<<")"<<endl;
                }
                break;
            }
            case 2: { // in array
                int start_idx = 0;
                int end_idx = -1;
                int index_end = 0;
                bool has_colon = false;
                while (ch != ']') {
                    if (ch == ':') {
                        buffer[index_end] = '\0';
                        start_idx = atoi(buffer);
                        end_idx = INT_MAX;
                        index_end = 0;
                        has_colon = true;
                    }
                    else if (ch >= '0' && ch <= '9') {
                        buffer[index_end++] = ch;
                    }
                    if (i + 1 == length) break;
                    ch = query[++i];
                }
                if (has_colon == false && index_end > 0) {
                    buffer[index_end] = '\0';
                    start_idx = atoi(buffer);
                    end_idx = start_idx + 1;
                } else if (index_end > 0) {
                    buffer[index_end] = '\0';
                    end_idx = atoi(buffer);
                }
                if (end_idx > -1) {
                    qa.addIndexConstraints(query_state, start_idx, end_idx);
                    // cout<<"index constraints "<<start_idx<<" "<<end_idx<<" current state "<<query_state<<endl;
                }
                if (i + 1 < length) {
                    ch = query[++i];
                    if (ch == '.') {
                        lexer_state = 1;
                        qa.updateStateTransInfo(query_state, false, NONE, OBJECT, NULL, query_state + 1);
                        // cout<<"("<<query_state<<", false, NONE, OBJECT, NULL, "<<(query_state + 1)<<")"<<endl;
                    } else if (ch == '[') {
                        cout<<"additional ["<<endl;
                        lexer_state = 2;
                        qa.updateStateTransInfo(query_state, false, NONE, ARRAY, NULL, query_state + 1);
                        // cout<<"("<<query_state<<", false, NONE, ARRAY, NULL, "<<(query_state + 1)<<")"<<endl;
                        ++query_state;
                    }
                } else {
                    qa.updateStateTransInfo(query_state, true, NONE, PRIMITIVE, NULL, query_state);
                    // cout<<"("<<query_state<<", false, NONE, PRIMITIVE, NULL, "<<(query_state + 1)<<")"<<endl;
                }
                break;
            }
        }
    }
}
