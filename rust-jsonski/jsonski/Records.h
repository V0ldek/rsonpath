#ifndef RECORDS_H
#define RECORDS_H

#include <iostream>
#include <stdlib.h>
#include <vector>
using namespace std;

#define MIN_RECORD_SIZE 5
#define MAX_RECORD_SIZE 1000000

// information for a single JSON record
struct Record {
    // for line-delimited JSON stream with a sequence of records,
    // contacting them into one single string generates the best
    // performance for indexing and querying
    char* text;
    long rec_start_pos;
    long rec_length;
    // text could be shared among different Record objects
    // (e.g. line-delimited JSON stream with a sequence of records)
    bool can_delete_text;

    Record() {
        text = NULL;
        rec_start_pos = 0;
        rec_length = 0;
        can_delete_text = true;
    }

    ~Record() {
        if (can_delete_text == true && text != NULL) {
            free(text);
            text = NULL;
            can_delete_text = false;
        }
    }
};

// information for a sequence of JSON records
class RecordSet {
    friend class RecordLoader;
  private:
    vector<Record*> recs;
    long num_recs;

  public:
    RecordSet() {
        num_recs = 0;
    }

    // record can be accessed in array style.
    Record*& operator[] (long idx) {
        if (idx >= 0 && idx < num_recs)
            return recs[idx];
        std::cerr << "Array index in RecordSet out of bound."<<endl; 
        exit(1); 
    }

    long size() {
        return num_recs;
    }

    ~RecordSet() {
        for (long i = 0; i < num_recs; ++i) {
            if (recs[i] != NULL)
                delete recs[i];
        }
    }
};
#endif
