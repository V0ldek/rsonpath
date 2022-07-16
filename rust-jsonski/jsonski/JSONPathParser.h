#ifndef JSONPATHPARSER_H
#define JSONPATHPARSER_H
#include <string.h>
#include "QueryAutomaton.h"

class JSONPathParser {
    public:
        // update query automaton based on the specific JSONPath query
        static void updateQueryAutomaton(string query, QueryAutomaton& qa);
};
#endif
