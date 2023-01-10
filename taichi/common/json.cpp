// Adapted from https://github.com/PENGUINLIONG/graphi-t

// Copyright (c) 2019 Rendong Liang
//
// Permission is hereby granted, free of charge, to any
// person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the
// Software without restriction, including without
// limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice
// shall be included in all copies or substantial portions
// of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
// ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
// TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
// PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
// SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

// JSON serialization/deserialization.
// @PENGUINLIONG
#include <sstream>
// #include "gft/log.hpp"
#include "taichi/common/json.h"

namespace liong {
namespace json {

enum JsonTokenType {
  L_JSON_TOKEN_UNDEFINED,
  L_JSON_TOKEN_NULL,
  L_JSON_TOKEN_TRUE,
  L_JSON_TOKEN_FALSE,
  L_JSON_TOKEN_STRING,
  L_JSON_TOKEN_INT,
  L_JSON_TOKEN_FLOAT,
  L_JSON_TOKEN_COLON,
  L_JSON_TOKEN_COMMA,
  L_JSON_TOKEN_OPEN_BRACE,
  L_JSON_TOKEN_CLOSE_BRACE,
  L_JSON_TOKEN_OPEN_BRACKET,
  L_JSON_TOKEN_CLOSE_BRACKET,
};
struct JsonToken {
  JsonTokenType ty;
  int64_t num_int;
  double num_float;
  std::string str;
};

struct Tokenizer {
  const char *pos;
  const char *end;

  Tokenizer(const char *beg, const char *end) : pos(beg), end(end) {
  }

  // Check the range first before calling this method.
  bool unsafe_starts_with(const char *head) {
    auto i = 0;
    while (*head != '\0') {
      if (pos[i++] != *(head++)) {
        return false;
      }
    }
    return true;
  }
  bool next_token(JsonToken &out) {
    std::stringstream ss;
    while (pos != end) {
      char c = *pos;

      if (c == '\0') {
        break;
      }

      // Ignore whitespaces.
      if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
        pos += 1;
        continue;
      }

      // Try parse scope punctuations.
      switch (c) {
        case ':':
          out.ty = L_JSON_TOKEN_COLON;
          pos += 1;
          return true;
        case ',':
          out.ty = L_JSON_TOKEN_COMMA;
          pos += 1;
          return true;
        case '{':
          out.ty = L_JSON_TOKEN_OPEN_BRACE;
          pos += 1;
          return true;
        case '}':
          out.ty = L_JSON_TOKEN_CLOSE_BRACE;
          pos += 1;
          return true;
        case '[':
          out.ty = L_JSON_TOKEN_OPEN_BRACKET;
          pos += 1;
          return true;
        case ']':
          out.ty = L_JSON_TOKEN_CLOSE_BRACKET;
          pos += 1;
          return true;
      }

      // Try parse numbers.
      if (c == '+' || c == '-' || (c >= '0' && c <= '9')) {
        out.ty = L_JSON_TOKEN_INT;
        const int STATE_INTEGRAL = 0;
        const int STATE_FRACTION = 1;
        const int STATE_EXPONENT = 2;
        int state = STATE_INTEGRAL;
        do {
          c = *pos;
          if (state == STATE_INTEGRAL) {
            if (c == '.') {
              state = STATE_FRACTION;
              ss.put(c);
              continue;
            }
            if (c == 'e') {
              state = STATE_EXPONENT;
              ss.put(c);
              continue;
            }
            if (c != '+' && c != '-' && (c < '0' || c > '9')) {
              break;
            }
          } else if (state == STATE_FRACTION) {
            out.ty = L_JSON_TOKEN_FLOAT;
            if (c == 'e') {
              state = STATE_EXPONENT;
              ss.put(c);
              continue;
            }
            if (c < '0' || c > '9') {
              break;
            }
          } else if (state == STATE_EXPONENT) {
            out.ty = L_JSON_TOKEN_FLOAT;
            if (c != '+' && c != '-' && (c < '0' || c > '9')) {
              break;
            }
          }
          ss.put(c);
        } while (++pos != end);
        if (out.ty == L_JSON_TOKEN_INT) {
          out.num_int = std::atoll(ss.str().c_str());
        } else if (out.ty == L_JSON_TOKEN_FLOAT) {
          out.num_float = std::atof(ss.str().c_str());
        }
        return true;
      }

      // Try parse strings.
      if (c == '"') {
        out.ty = L_JSON_TOKEN_STRING;
        bool escape = false;
        while (++pos != end) {
          c = *pos;
          if (escape) {
            switch (c) {
              case '"':
              case '/':
                break;
              case 'b':
                c = '\b';
                break;
              case 'f':
                c = '\f';
                break;
              case 'n':
                c = '\n';
                break;
              case 'r':
                c = '\r';
                break;
              case 't':
                c = '\t';
                break;
              case 'u':
                throw JsonException("unicode escape is not supported");
              default:
                throw JsonException("invalid escape charater");
            }
            escape = false;
          } else {
            if (c == '\\') {
              escape = true;
              continue;
            } else if (c == '"') {
              if (escape != false) {
                throw JsonException("incomplete escape sequence");
              }
              out.str = ss.str();
              pos += 1;
              return true;
            }
          }
          ss.put(c);
        }
        throw JsonException("unexpected end of string");
      }

      // Try parse literals.
      if (pos + 4 <= end) {
        if (unsafe_starts_with("null")) {
          out.ty = L_JSON_TOKEN_NULL;
          pos += 4;
          return true;
        }
        if (unsafe_starts_with("true")) {
          out.ty = L_JSON_TOKEN_TRUE;
          pos += 4;
          return true;
        }
      }
      if (pos + 5 <= end) {
        if (unsafe_starts_with("false")) {
          out.ty = L_JSON_TOKEN_FALSE;
          pos += 5;
          return true;
        }
      }
    }
    out.ty = L_JSON_TOKEN_UNDEFINED;
    return false;
  }
};

bool try_parse_impl(Tokenizer &tokenizer, JsonValue &out) {
  JsonToken token;
  while (tokenizer.next_token(token)) {
    JsonValue val;
    switch (token.ty) {
      case L_JSON_TOKEN_TRUE:
        out.ty = L_JSON_BOOLEAN;
        out.b = true;
        return true;
      case L_JSON_TOKEN_FALSE:
        out.ty = L_JSON_BOOLEAN;
        out.b = false;
        return true;
      case L_JSON_TOKEN_NULL:
        out.ty = L_JSON_NULL;
        return true;
      case L_JSON_TOKEN_STRING:
        out.ty = L_JSON_STRING;
        out.str = std::move(token.str);
        return true;
      case L_JSON_TOKEN_INT:
        out.ty = L_JSON_INT;
        out.num_int = token.num_int;
        return true;
      case L_JSON_TOKEN_FLOAT:
        out.ty = L_JSON_FLOAT;
        out.num_int = token.num_float;
        return true;
      case L_JSON_TOKEN_OPEN_BRACKET:
        out.ty = L_JSON_ARRAY;
        for (;;) {
          if (!try_parse_impl(tokenizer, val)) {
            // When the array has no element.
            break;
          }
          out.arr.inner.emplace_back(std::move(val));
          if (tokenizer.next_token(token)) {
            if (token.ty == L_JSON_TOKEN_COMMA) {
              continue;
            } else if (token.ty == L_JSON_TOKEN_CLOSE_BRACKET) {
              break;
            } else {
              throw JsonException("unexpected token in array");
            }
          } else {
            throw JsonException("unexpected end of array");
          }
        }
        return true;
      case L_JSON_TOKEN_OPEN_BRACE:
        out.ty = L_JSON_OBJECT;
        for (;;) {
          // Match the key.
          std::string key;
          if (tokenizer.next_token(token)) {
            if (token.ty == L_JSON_TOKEN_STRING) {
              key = std::move(token.str);
            } else if (token.ty == L_JSON_TOKEN_CLOSE_BRACE) {
              // The object has no field.
              break;
            } else {
              throw JsonException("unexpected object field key type");
            }
          } else {
            throw JsonException("unexpected end of object");
          }
          // Match the colon.
          if (!tokenizer.next_token(token)) {
            throw JsonException("unexpected end of object");
          }
          if (token.ty != L_JSON_TOKEN_COLON) {
            throw JsonException("unexpected token in object");
          }
          // Match the value.
          if (!try_parse_impl(tokenizer, val)) {
            throw JsonException("unexpected end of object");
          }
          out.obj.inner[key] = std::move(val);
          // Should we head for another round?
          if (tokenizer.next_token(token)) {
            if (token.ty == L_JSON_TOKEN_COMMA) {
              continue;
            } else if (token.ty == L_JSON_TOKEN_CLOSE_BRACE) {
              break;
            } else {
              throw JsonException("unexpected token in object");
            }
          } else {
            throw JsonException("unexpected end of object");
          }
        }
        return true;
      case L_JSON_TOKEN_CLOSE_BRACE:
      case L_JSON_TOKEN_CLOSE_BRACKET:
        return false;
      default:
        throw JsonException("unexpected token");
    }
  }
  throw JsonException("unexpected program state");
}

JsonValue parse(const char *beg, const char *end) {
  if (beg == nullptr || end == nullptr || beg >= end) {
    throw JsonException("json text is empty");
  }
  JsonValue rv;
  Tokenizer tokenizer(beg, end);
  if (!try_parse_impl(tokenizer, rv)) {
    throw JsonException("unexpected close token");
  }
  return rv;
}
JsonValue parse(const std::string &json_lit) {
  return parse(json_lit.c_str(), json_lit.c_str() + json_lit.size());
}
bool try_parse(const std::string &json_lit, JsonValue &out) {
  try {
    out = parse(json_lit);
  } catch (JsonException e) {
    // log::error("failed to parse json: ", e.what());
    return true;
  }
  return false;
}

void print_impl(const JsonValue &json, std::stringstream &out) {
  switch (json.ty) {
    case L_JSON_NULL:
      out << "null";
      return;
    case L_JSON_BOOLEAN:
      out << (json.b ? "true" : "false");
      return;
    case L_JSON_FLOAT:
      out << json.num_float;
      return;
    case L_JSON_INT:
      out << json.num_int;
      return;
    case L_JSON_STRING:
      out << "\"" << json.str << "\"";
      return;
    case L_JSON_OBJECT:
      out << "{";
      {
        bool is_first_iter = true;
        for (const auto &pair : json.obj.inner) {
          if (is_first_iter) {
            is_first_iter = false;
          } else {
            out << ",";
          }
          out << "\"" << pair.first << "\":";
          print_impl(pair.second, out);
        }
      }
      out << "}";
      return;
    case L_JSON_ARRAY:
      out << "[";
      {
        bool is_first_iter = true;
        for (const auto &elem : json.arr.inner) {
          if (is_first_iter) {
            is_first_iter = false;
          } else {
            out << ",";
          }
          print_impl(elem, out);
        }
      }
      out << "]";
      return;
  }
}
std::string print(const JsonValue &json) {
  std::stringstream ss;
  print_impl(json, ss);
  return ss.str();
}

}  // namespace json
}  // namespace liong
