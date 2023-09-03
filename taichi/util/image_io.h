#pragma once

#include <string>
#include <vector>

namespace taichi {

void imwrite(const std::string &filename,
             size_t ptr,
             int resx,
             int resy,
             int comp);
std::vector<size_t> imread(const std::string &filename, int comp);

}  // namespace taichi
