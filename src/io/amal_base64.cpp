#include <taichi/io/base64.h>
#include <taichi/common/task.h>

TC_NAMESPACE_BEGIN

auto amal_base64 = [](const std::vector<std::string> &param) {
  TC_ASSERT(param.size() >= 1);
  auto fn = param[0];
  std::ifstream input(fn);
  std::string str((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  auto encoded = base64_encode(str);
  if(param.size() >= 2) {
    auto line_width = 78;
    auto fo = fopen(param[2].c_str(), "w");
    auto maximum_literal_length = 65500 / line_width * line_width; // MSVC cannot deal with literal with length > 65535
    fmt::print(fo, "#include <taichi/common/util.h>\n\nTC_NAMESPACE_BEGIN\n\n\n");
    int num_literals = 0;
    for (int l = 0; l < encoded.size(); l += maximum_literal_length) {
      fmt::print(fo, "const std::string {}_{:04d} =\n", param[1], num_literals);
      num_literals += 1;
      for (int i = l; i < std::min((int)encoded.size(), l + maximum_literal_length); i += line_width) {
        fmt::print(fo, "\"{}\"\n", encoded.substr(i, line_width));
      }
      fmt::print(fo, ";\n");
    }

    fmt::print(fo, "const std::string {} = ", param[1]);
    for (int i = 0; i < num_literals; i++) {
      fmt::print(fo, "{}_{:04d}", param[1], i);
      if (i < num_literals - 1)
        fmt::print(fo, " + ", param[1], i);
    }

    fmt::print(fo, ";\n\nTC_NAMESPACE_END");
    std::fclose(fo);
  }
};

TC_REGISTER_TASK(amal_base64);

TC_NAMESPACE_END
