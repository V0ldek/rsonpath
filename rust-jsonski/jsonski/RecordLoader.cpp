#include <sys/time.h>
#include "RecordLoader.h"
using namespace std;

#define MAX_PAD 64

Record* RecordLoader::loadSingleRecord(char* file_path) {
    unsigned long size;
    FILE* fp = fopen (file_path,"rb");
    if (fp == NULL) {
        return NULL;
    }
    fseek (fp, 0, SEEK_END);
    size = ftell(fp);
    rewind(fp);
    void* p;
    if (posix_memalign(&p, 64, (size + MAX_PAD)*sizeof(char)) != 0) {
        cout<<"Fail to allocate memory space for input record."<<endl;
    }
    char* record_text = (char*) p;
    size_t load_size = fread (record_text, 1, size, fp);
    if (load_size == 0) {
        cout<<"Fail to load the input record into memory"<<endl;
    }
    int remain = 64 - (size % 64);
    int counter = 0;
    // pad the input data where its size can be divided by 64
    while (counter < remain)
    {
        record_text[size+counter] = 'd';
        counter++;
    }
    record_text[size+counter]='\0';
    fclose(fp);
    // only one single record
    Record* record = new Record();
    record->text = record_text;
    record->rec_start_pos = 0;
    record->rec_length = strlen(record_text);
    return record;
}

RecordSet* RecordLoader::loadRecords(char* file_path) {
    FILE *fp = fopen(file_path, "r");
    RecordSet* rs = new RecordSet();
    if (fp) {
        char line[MAX_RECORD_SIZE];
        string str;
        int start_pos = 0;
        while (fgets(line, sizeof(line), fp) != NULL) {
            if (strlen(line) <= MIN_RECORD_SIZE) continue;
            int remain = 64 - strlen(line) % 64;
            int top = strlen(line);
            while (remain > 0) {
                line[top++] = 'd';
                --remain;
            }
            line[top] = '\0';
            if (strlen(line) > MIN_RECORD_SIZE) {
                // concating a sequence of record texts into one single string generates the best performance for indexing and querying
                str.append(line);
                Record* record = new Record();
                record->rec_start_pos = start_pos;
                record->rec_length = strlen(line);
                start_pos += strlen(line);
                rs->recs.push_back(record);
                ++rs->num_recs;
            }
        }
        void* p;
        if(posix_memalign(&p, 64, str.size()*sizeof(char)) != 0) {
            cout<<"Fail to allocate memory space for records from input file."<<endl;
        }
        for (int i = 0; i < rs->recs.size(); ++i) {
            // all record objects points to the same input text which contacts a sequence of JSON records
            rs->recs[i]->text = (char*) p;
            if (i == 0) strcpy(rs->recs[0]->text, str.c_str());
            // deconstructor in the last record object can delete input text
            if (i < rs->recs.size() - 1) rs->recs[i]->can_delete_text = false;
        }
        fclose(fp);
        return rs;
    }
    cout<<"Fail open the file."<<endl;
    return rs;
}
