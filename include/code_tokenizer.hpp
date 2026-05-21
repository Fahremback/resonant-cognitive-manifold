#pragma once
#include <string>
#include <vector>

namespace rcm {

class CodeTokenizer {
public:
    static std::vector<std::string> tokenize(const std::string& code);
    static std::string detokenize(const std::vector<std::string>& tokens);
};

} // namespace rcm
