#include "short_name.h"
#include "base64.h"

const std::string alphatable = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"; // e.g. `if` is not a good var name, but `If` does.
const std::string alnumtable = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

std::string make_short_name_by_id(int id)
{
  std::string res;
  while (id >= alphatable.size()) {
    res.insert(0, 1, alnumtable[id % alnumtable.size()]);
    id /= alnumtable.size();
  }
  res.insert(0, 1, alphatable[id]);
  return res;
}
