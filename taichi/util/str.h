#pragma once

#include <string>
#include <functional>

#include "taichi/lang_util.h"

TLANG_NAMESPACE_BEGIN

// Quote |str| with a pair of ". Escape special characters like \n, \t etc.
std::string c_quoted(std::string const &str);

std::string format_error_message(const std::string &error_message_template,
                                 const std::function<uint64(int)> &fetcher);

TLANG_NAMESPACE_END
