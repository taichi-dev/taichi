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
    fmt::print(fo, "#include <taichi/common/util.h>\n\nTC_NAMESPACE_BEGIN\n\nconst std::string {} = \n", param[1]);
    for (int i = 0; i < encoded.size(); i+= line_width) {
      fmt::print(fo, "\"{}\"\n", encoded.substr(i, line_width));
    }
    fmt::print(fo, ";\n\nTC_NAMESPACE_END");
    std::fclose(fo);
  }
};

TC_REGISTER_TASK(amal_base64);

TC_NAMESPACE_END
