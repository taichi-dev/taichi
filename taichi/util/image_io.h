#pragma once

#include <string>
#include <vector>

TI_NAMESPACE_BEGIN
void tc_imwrite(const std::string &filename, size_t ptr, int resx, int resy, int comp);
std::vector<size_t> tc_imread(const std::string &filename, int comp);
TI_NAMESPACE_END
