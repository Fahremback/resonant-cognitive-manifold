#include "code_tokenizer.hpp"
#include <cctype>
#include <iostream>

namespace rcm {

std::vector<std::string> CodeTokenizer::tokenize(const std::string& code) {
    std::vector<std::string> tokens;
    size_t i = 0;
    size_t length = code.length();
    bool is_start_of_line = true;

    while (i < length) {
        // Trata recuo no início da linha
        if (is_start_of_line) {
            size_t space_count = 0;
            while (i < length && (code[i] == ' ' || code[i] == '\t')) {
                space_count++;
                i++;
            }
            if (space_count > 0) {
                tokens.push_back(std::string(space_count, ' '));
            }
            is_start_of_line = false;
            continue;
        }

        // Ignora espaços simples de separação fora do início da linha
        if (code[i] == ' ' || code[i] == '\t') {
            i++;
            continue;
        }

        // Ignora comentários do Python
        if (code[i] == '#') {
            while (i < length && code[i] != '\n') {
                i++;
            }
            continue;
        }

        // Trata quebras de linha
        if (code[i] == '\n') {
            tokens.push_back("\n");
            is_start_of_line = true;
            i++;
            continue;
        }
        if (code[i] == '\r') {
            i++;
            continue;
        }

        // Trata strings literais
        if (code[i] == '\'' || code[i] == '"') {
            char quote = code[i];
            std::string literal = "";
            literal += quote;
            i++;
            while (i < length && code[i] != quote) {
                if (code[i] == '\\' && i + 1 < length) {
                    literal += '\\';
                    literal += code[i + 1];
                    i += 2;
                } else {
                    literal += code[i];
                    i++;
                }
            }
            if (i < length && code[i] == quote) {
                literal += quote;
                i++;
            }
            tokens.push_back(literal);
            continue;
        }

        // Operadores de 2 caracteres
        if (i + 1 < length) {
            std::string op2 = code.substr(i, 2);
            if (op2 == "==" || op2 == "!=" || op2 == "<=" || op2 == ">=" ||
                op2 == "+=" || op2 == "-=" || op2 == "*=" || op2 == "/=" || 
                op2 == "//") {
                tokens.push_back(op2);
                i += 2;
                continue;
            }
        }

        // Números e identificadores
        if (std::isalnum(static_cast<unsigned char>(code[i])) || code[i] == '_') {
            std::string id = "";
            while (i < length && (std::isalnum(static_cast<unsigned char>(code[i])) || code[i] == '_')) {
                id += code[i];
                i++;
            }
            tokens.push_back(id);
            continue;
        }

        // Operadores e símbolos de 1 caractere
        std::string op1 = "";
        op1 += code[i];
        tokens.push_back(op1);
        i++;
    }

    return tokens;
}

std::string CodeTokenizer::detokenize(const std::vector<std::string>& tokens) {
    std::string code = "";
    for (size_t i = 0; i < tokens.size(); ++i) {
        const std::string& token = tokens[i];
        
        if (token == "\n") {
            code += "\n";
            continue;
        }

        // Se for o início do código ou após uma quebra de linha, ou se o token atual for espaço de recuo, não adiciona espaço prévio
        bool after_newline = (i > 0 && tokens[i - 1] == "\n") || i == 0;
        bool is_indent = !token.empty() && token[0] == ' ';

        if (i > 0 && !after_newline && !is_indent) {
            const std::string& prev = tokens[i - 1];
            // Regras para decidir se precisa de espaço antes do token
            // Não colocamos espaço após operadores de ponto '.', parênteses abrindo '(', etc.
            if (prev != "." && prev != "(" && prev != "[" && prev != "{" &&
                token != "." && token != ")" && token != "]" && token != "}" && 
                token != ":" && token != "," && token != "(" && prev != "    " && prev != "        ") {
                code += " ";
            }
        }

        code += token;
    }
    return code;
}

} // namespace rcm
