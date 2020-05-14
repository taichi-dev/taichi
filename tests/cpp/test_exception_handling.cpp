#include "taichi/common/task.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"

TLANG_NAMESPACE_BEGIN

// Simulate the AST transforms on the Google Colab Ubuntu kernels to see why it
// crashes

class NodeBase {
 public:
  std::unique_ptr<NodeBase> ch;

  NodeBase(std::unique_ptr<NodeBase> &&ch) : ch(std::move(ch)) {
  }

  virtual void visit() = 0;

  virtual ~NodeBase() = default;
};

class NodeA : public NodeBase {
 public:
  NodeA(std::unique_ptr<NodeBase> &&ch) : NodeBase(std::move(ch)) {
  }

  void visit() override {
    TI_INFO("Visiting node A");
    if (ch)
      ch->visit();
  }
};

class NodeB : public NodeBase {
 public:
  NodeB(std::unique_ptr<NodeBase> &&ch) : NodeBase(std::move(ch)) {
  }

  void visit() override {
    TI_INFO("Visiting node B, throwing std::exception");
    throw std::exception();
  }
};

class NodeC : public NodeBase {
 public:
  NodeC(std::unique_ptr<NodeBase> &&ch) : NodeBase(std::move(ch)) {
  }

  void visit() override {
    TI_INFO("Visiting node C, throwing IRModified");
    throw IRModified();
  }
};

int test_throw(const std::string &seq) {
  TI_ASSERT(seq.size() >= 0);
  std::unique_ptr<NodeBase> root;
  TI_P(seq);
  for (int i = (int)seq.size() - 1; i >= 0; i--) {
    auto ch = seq[i];
    if (ch == 'A') {
      root = std::make_unique<NodeA>(std::move(root));
    } else if (ch == 'B') {
      root = std::make_unique<NodeB>(std::move(root));
    } else if (ch == 'C') {
      root = std::make_unique<NodeC>(std::move(root));
    } else {
      TI_NOT_IMPLEMENTED;
    }
  }
  try {
    root->visit();
  } catch (const IRModified &) {
    TI_INFO("Caught IRModified (Node C)");
    return 2;
  } catch (const std::exception &) {
    TI_INFO("Caught std::exception (Node B)");
    return 1;
  }
  return 0;
}

auto test_exception_handling = [](const std::vector<std::string> &params) {
  test_throw(params[0]);
};

auto test_exception_handling_auto = []() {
  TI_ASSERT(test_throw("A") == 0);
  TI_ASSERT(test_throw("AAA") == 0);
  TI_ASSERT(test_throw("AAB") == 1);
  TI_ASSERT(test_throw("AAC") == 2);
  TI_ASSERT(test_throw("AACB") == 2);
  TI_ASSERT(test_throw("AABC") == 1);

  TI_INFO("Test was successful");
};

TI_REGISTER_TASK(test_exception_handling);
TI_REGISTER_TASK(test_exception_handling_auto);

TLANG_NAMESPACE_END
