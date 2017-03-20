/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

struct DCRawOutput {
    int width, height, channels;
    float *data = nullptr;
    void initialize(int width, int height, int channels) {
        this->width = width;
        this->height = height;
        this->channels = channels;
        data = new float[width * height * channels];
    }
};

int dcraw_main(int argc, const char **argv, DCRawOutput &output);

