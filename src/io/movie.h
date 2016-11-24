#pragma once

#include "common/utils.h"
#include "visualization/image_buffer.h"

TC_NAMESPACE_BEGIN

typedef ImageBuffer<Vector3> MovieFrame;

class Movie {
private:
	std::vector<MovieFrame> frames;
public:
	MovieFrame &get_frame(int frame_number) {
		assert_info(0 <= frame_number && frame_number < (int)frames.size(), string("Frame Number out of bound!"));
		return frames[frame_number];
	}
	int get_num_frames() {
		return (int)frames.size();
	}
	void push_frame(const MovieFrame &frame) {
		frames.push_back(frame);
	}
	std::vector<MovieFrame>::const_iterator get_frame_iterator() {
		return frames.begin();
	}
};

TC_NAMESPACE_END

