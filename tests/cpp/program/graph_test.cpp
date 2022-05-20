#include "gtest/gtest.h"
#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
#include "taichi/inc/constants.h"
#include "taichi/program/program.h"
#include "tests/cpp/program/test_program.h"
#include "taichi/program/graph.h"
#include "tests/cpp/ir/ndarray_kernel.h"
#ifdef TI_WITH_VULKAN
#include "taichi/backends/vulkan/vulkan_loader.h"
#endif

using namespace taichi;
using namespace lang;
#ifdef TI_WITH_VULKAN
TEST(GraphTest, SimpleGraphRun) {
  // Otherwise will segfault on macOS VM,
  // where Vulkan is installed but no devices are present
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }
  TestProgram test_prog;
  test_prog.setup(Arch::vulkan);

  const int size = 10;

  auto ker1 = setup_kernel1(test_prog.prog());
  auto ker2 = setup_kernel2(test_prog.prog());

  auto g = std::make_unique<Graph>("test");
  auto seq = g->seq();
  auto arr_arg = aot::Arg{
      aot::ArgKind::NDARRAY, "arr", PrimitiveType::i32.to_string(), {size}};
  seq->emplace(ker1.get(), {arr_arg});
  seq->emplace(ker2.get(), {arr_arg, aot::Arg{
                                         aot::ArgKind::SCALAR,
                                         "x",
                                         PrimitiveType::i32.to_string(),
                                     }});
  g->compile();

  auto array = Ndarray(test_prog.prog(), PrimitiveType::i32, {size});
  array.write_int({0}, 2);
  array.write_int({2}, 40);
  std::unordered_map<std::string, aot::IValue> args;
  args.insert({"arr", aot::IValue::create(array)});
  args.insert({"x", aot::IValue::create<int>(2)});

  g->run(args);
  EXPECT_EQ(array.read_int({0}), 2);
  EXPECT_EQ(array.read_int({1}), 2);
  EXPECT_EQ(array.read_int({2}), 42);
}
#endif
