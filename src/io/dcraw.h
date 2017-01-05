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

