#include "util.h"

TC_NAMESPACE_BEGIN

namespace zip {

void write(std::string fn, const uint8 *data, std::size_t len);
void write(const std::string &fn, const std::string &data);
std::vector<uint8> read(const std::string fn, bool verbose = false);

}  // namespace zip

TC_NAMESPACE_END
