#include "short_name.h"
#include "base64.h"

TI_NAMESPACE_BEGIN
namespace {
const std::string alphatable =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ";  // e.g. `if` is not a good var name, but `If`
                                   // does.
const std::string alnumtable =
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
};  // namespace

std::string make_short_name_by_id(int id) {
  std::string res;
  while (id >= alphatable.size()) {
    res.push_back(alnumtable[id % alnumtable.size()]);
    id /= alnumtable.size();
  }
  res.push_back(alphatable[id]);
  std::reverse(res.begin(), res.end());
  return res;
}
TI_NAMESPACE_END
