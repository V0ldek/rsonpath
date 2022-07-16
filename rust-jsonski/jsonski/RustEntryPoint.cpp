#include "RustEntryPoint.h"
#include "RecordLoader.h"
#include "QueryProcessor.h"
#include <iostream>

void *loadFile(char *file_path)
{
    Record *rec = RecordLoader::loadSingleRecord(file_path);

    if (rec == NULL)
    {
        std::cerr << "panic: failed to load record\n";
        exit(1);
    }

    return rec;
}

long runJsonSki(char *query, Record *record)
{
    QueryProcessor processor(query);

    return processor.runQuery(record);
}