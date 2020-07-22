#include <iostream>
#include <fstream>
#include <cassert>
#include <string>
#include <vector>
#include <map>

using ActionEntry = std::map<std::string, std::string>;

class ActionParser {
 public:
  ActionParser(std::istream &is) : is(is) {}
  ActionEntry parse();

 private:
  std::string toke_str();
  void skip_space();
  int next_line();

  static char c_unquote(char c);

  std::istream &is;
};

char ActionParser::c_unquote(char c) {
  // https://zh.cppreference.com/w/cpp/language/escape
  switch (c) {
#define REG_ESC(x, y) \
case y:             \
  return x;
    REG_ESC('\n', 'n');
    REG_ESC('\a', 'a');
    REG_ESC('\b', 'b');
    REG_ESC('\?', '?');
    REG_ESC('\v', 'v');
    REG_ESC('\t', 't');
    REG_ESC('\f', 'f');
    REG_ESC('\'', '\'');
    REG_ESC('\"', '\"');
    REG_ESC('\\', '\\');
  default:
    assert(0 && c);
  }
#undef REG_ESC
}

std::string ActionParser::toke_str() {
  std::string ret;
  char c;
  is.get(c);
  if (c != '"') {
    is.unget();
    while (1) {
      is.get(c);
      if (!isalnum(c) && c != '_') {
        is.unget();
        break;
      }
      ret.push_back(c);
    }
    return ret;
  }

  while (1) {
    is.get(c);
    if (c == '\\') {
      is.get(c);
      c = c_unquote(c);
    } else if (c == '"')
      break;
    ret.push_back(c);
  }

  return ret;
}


void ActionParser::skip_space() {
  char c;
  do {
    is.get(c);
  } while (c == ' ');

  is.unget();
}


ActionEntry ActionParser::parse() {
  ActionEntry ret;
  char c;

  is.get(c);
  assert(c == '-');

  while (1) {
    skip_space();
    is >> c;
    is.unget();

    if (!is || c == '-')
      break;

    auto key = toke_str();
    is.get(c);
    assert(c == ':');
    skip_space();
    auto value = toke_str();
    ret[key] = value;
  }

  return ret;
}


int main(void)
{
  std::ifstream ifs("record.yml");
  ActionParser ap(ifs);
  ActionEntry ae;
  ae = ap.parse();
  std::cout << ae["action"] << std::endl;
  std::cout << ae["layout_source"] << std::endl;
  ae = ap.parse();
  std::cout << ae["action"] << std::endl;
  std::cout << ae["root_size"] << std::endl;
  std::cout << ae["gtmp_size"] << std::endl;
  ae = ap.parse();
  std::cout << ae["action"] << std::endl;
  std::cout << ae["kernel_name"] << std::endl;
  std::cout << ae["kernel_source"] << std::endl;
  ae = ap.parse();
  std::cout << ae["action"] << std::endl;
  std::cout << ae["kernel_name"] << std::endl;
  return 0;
}
