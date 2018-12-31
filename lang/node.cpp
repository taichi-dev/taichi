#include "node.h"
#include "expr.h"

TLANG_NAMESPACE_BEGIN

int Node::counter = 0;
std::map<Node::NodeType, std::string> Node::node_type_names;

int Node::group_size() const {
  return (int)members.size();
}

TLANG_NAMESPACE_END
