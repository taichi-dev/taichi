#define _DEBUG
#include <iostream>
#include <fstream>
#include <cassert>
#include <string>
#include <vector>
#include <map>

using ActionEntry = std::map<std::string, std::string>;

class ActionParser {
 public:
  ActionParser(std::istream &is) : is(is) {
  }
  ActionEntry parse();
  bool iseof();

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
      assert(0 && "undefined escape sequence");
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
      if (!isalnum(c) && c != '_' && c != '-') {
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

bool ActionParser::iseof() {
  return !is;
}

ActionEntry ActionParser::parse() {
  assert(is && "already eof");

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

#define ACTION_LIST            \
  REG_ACTION(compile_runtime); \
  REG_ACTION(compile_layout);  \
  REG_ACTION(allocate_buffer); \
  REG_ACTION(compile_kernel);  \
  REG_ACTION(launch_kernel);   \
  ;

class ActionExecuter {
 public:
  void run(std::ifstream &ifs);

 private:
  void execute_action(ActionEntry const &ae);

 protected:
#define REG_ACTION(name) virtual void do_##name(ActionEntry const &ae);
  ACTION_LIST;
#undef REG_ACTION
};

#define REG_ACTION(name)                                  \
  void ActionExecuter::do_##name(ActionEntry const &ae) { \
    assert(0 && "unhandled action " #name);               \
  }
ACTION_LIST;
#undef REG_ACTION

void ActionExecuter::execute_action(ActionEntry const &ae) {
  auto action = ae.at("action");
  if (0) {
#define REG_ACTION(name)      \
  }                           \
  else if (action == #name) { \
    do_##name(ae);
    ACTION_LIST;
  }
#undef REG_ACTION
}

void ActionExecuter::run(std::ifstream &ifs) {
  ActionParser ap(ifs);
  while (!ap.iseof()) {
    auto ae = ap.parse();
    execute_action(ae);
  }
}

class ActionExecuterCCSave : public ActionExecuter {
 public:
  ActionExecuterCCSave(std::ostream &os) : os(os) {
  }

  void make_main();

 private:
  std::ostream &os;

 protected:
#define REG_ACTION(name) virtual void do_##name(ActionEntry const &ae) override;
  ACTION_LIST;
#undef REG_ACTION

  std::vector<ActionEntry> launches;
};

void ActionExecuterCCSave::do_compile_runtime(ActionEntry const &ae) {
  auto header = ae.at("runtime_header");
  auto source = ae.at("runtime_source");
  os << header;
  os << source;
}

void ActionExecuterCCSave::do_compile_layout(ActionEntry const &ae) {
  auto source = ae.at("layout_source");
  os << source;
  os << "struct Ti_S0root Ti_root;\n";
}

void ActionExecuterCCSave::do_allocate_buffer(ActionEntry const &ae) {
  auto root_size = ::atoi(ae.at("root_size").c_str());
  auto gtmp_size = ::atoi(ae.at("gtmp_size").c_str());
  os << "Ti_i8 Ti_gtmp[" << gtmp_size << "];\n";
  os << "union Ti_BitCast Ti_args[8];\n";
}

void ActionExecuterCCSave::do_compile_kernel(ActionEntry const &ae) {
  auto name = ae.at("kernel_name");
  auto source = ae.at("kernel_source");
  os << source;
}

void ActionExecuterCCSave::do_launch_kernel(ActionEntry const &ae) {
  launches.push_back(ae);
}

void ActionExecuterCCSave::make_main() {
  os << "int main(void) {\n";
  os << "  struct Ti_Context ti_ctx;\n";
  os << "  ti_ctx.root = &Ti_root;\n";
  os << "  ti_ctx.gtmp = Ti_gtmp;\n";
  os << "  ti_ctx.args = Ti_args;\n";
  for (auto const &ae : launches) {
    auto name = ae.at("kernel_name");
    os << "  Tk_" << name << "(&ti_ctx);\n";
  }
  os << "}\n";
}

int main(int argc, char **argv) {
#ifdef _DEBUG
  std::ifstream is("record.yml");
  auto &os = std::cout;
  bool has_main = true;
#else
  if (argc <= 2) {
    std::cerr << "usage: " << argv[0] << " <input.yml> <output.c> [-m]"
              << std::endl;
    return 1;
  }
  std::ifstream is(argv[1]);
  std::ifstream os(argv[2]);
  bool has_main = (argv[3] && std::string(argv[3]) == "-m";
#endif

  ActionExecuterCCSave ax(os);
  ax.run(is);
  if (has_main) {
    ax.make_main();
  }
  return 0;
}
