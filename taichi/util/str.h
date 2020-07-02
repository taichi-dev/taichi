#pragma once

#include <string>

#include "taichi/lang_util.h"

TLANG_NAMESPACE_BEGIN

// Quote |str| with a pair of ". Escape special characters like \n, \t etc.
std::string c_quoted(std::string const &str);

TLANG_NAMESPACE_END
