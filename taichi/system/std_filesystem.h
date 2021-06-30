#pragma once

#if __has_include(<filesystem>)
#include <filesystem>
namespace stdfs = ::std::filesystem;
#else
#include <experimental/filesystem>
namespace stdfs = ::std::experimental::filesystem;
#endif
