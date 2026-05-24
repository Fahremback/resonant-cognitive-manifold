#pragma once
#include <vector>
#include <string>
#include <unordered_map>

namespace rcm {

struct Token {
    int id;
    std::string text;
    float position;
};

class CodeTokenizer {
public:
    CodeTokenizer();
    std::vector<Token> tokenize(const std::string& code);
    std::string detokenize(const std::vector<Token>& tokens);
    void add_special_tokens(const std::vector<std::string>& tokens);
    
private:
    std::unordered_map<std::string, int> token_to_id;
    std::unordered_map<int, std::string> id_to_token;
    std::vector<std::string> special_tokens;
    int next_id;
};

} // namespace rcm
