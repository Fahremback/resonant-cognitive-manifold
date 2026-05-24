#include "code_tokenizer.hpp"
#include <sstream>
#include <regex>

namespace rcm {

CodeTokenizer::CodeTokenizer() : next_id(0) {
    // Tokens especiais padrão
    add_special_tokens({"<PAD>", "<UNK>", "<SOS>", "<EOS>", "<SPACE>", "<TAB>", "<NEWLINE>"});
}

void CodeTokenizer::add_special_tokens(const std::vector<std::string>& tokens) {
    for (const auto& token : tokens) {
        if (token_to_id.find(token) == token_to_id.end()) {
            token_to_id[token] = next_id;
            id_to_token[next_id] = token;
            special_tokens.push_back(token);
            next_id++;
        }
    }
}

std::vector<Token> CodeTokenizer::tokenize(const std::string& code) {
    std::vector<Token> tokens;
    
    // Regex simplificado para tokenização de código
    std::regex word_regex(R"(\b\w+\b|[{}()\[\];,.]|\"[^\"]*\"|\'[^\']*\'|\s+)");
    auto words_begin = std::sregex_iterator(code.begin(), code.end(), word_regex);
    auto words_end = std::sregex_iterator();
    
    float position = 0.0f;
    for (auto i = words_begin; i != words_end; ++i) {
        std::string match = i->str();
        
        Token token;
        token.text = match;
        token.position = position;
        
        if (token_to_id.find(match) != token_to_id.end()) {
            token.id = token_to_id[match];
        } else {
            // Token desconhecido ou novo
            token.id = token_to_id.count("<UNK>") > 0 ? token_to_id["<UNK>"] : next_id++;
            if (token_to_id.find(match) == token_to_id.end()) {
                token_to_id[match] = token.id;
                id_to_token[token.id] = match;
            }
        }
        
        tokens.push_back(token);
        position += 1.0f;
    }
    
    return tokens;
}

std::string CodeTokenizer::detokenize(const std::vector<Token>& tokens) {
    std::stringstream ss;
    for (const auto& token : tokens) {
        ss << token.text;
    }
    return ss.str();
}

} // namespace rcm
