/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <string>
#include <vector>
#include "util.h"

TC_NAMESPACE_BEGIN

    inline std::vector<std::string> split_string(const std::string &s, const std::string &seperators) {
        std::vector<std::string> ret;
        bool is_seperator[256] = {false};
        for (auto &ch: seperators) {
            is_seperator[(unsigned int) ch] = true;
        }
        int begin = 0;
        for (int i = 0; i <= (int) s.size(); i++) {
            if (is_seperator[s[i]] || i == (int) s.size()) {
                ret.push_back(std::string(s.begin() + begin, s.begin() + i));
                begin = i + 1;
            }
        }
        return ret;
    }

    inline std::string trim_string(const std::string &s) {
        int begin = 0, end = (int) s.size();
        while (begin < end && s[begin] == ' ') {
            begin++;
        }
        while (begin < end && s[end - 1] == ' ') {
            end--;
        }
        return std::string(s.begin() + begin, s.begin() + end);
    }


TC_NAMESPACE_END
