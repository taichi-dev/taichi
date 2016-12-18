#pragma once

#include <taichi/math/linalg.h>
#include <taichi/common/meta.h>
#include <vector>
#include <memory>
#include <string>

TC_NAMESPACE_BEGIN

class StateSequence {
protected:
	int cursor = 0;
public:
	virtual real sample() = 0;

	virtual real operator()() {
		return sample();
	}

	int get_cursor() const {
		return cursor;
	}

	void assert_cursor_pos(int cursor) const {
		assert_info(this->cursor == cursor, std::string("Cursor position should be " + std::to_string(cursor) +
			" instead of " + std::to_string(this->cursor)));
	}
};


class Sampler {
public:
	virtual void initialize(const Config &config) {};
	virtual real sample(int d, long long i) const = 0;
};
TC_INTERFACE(Sampler);

class RandomStateSequence : public StateSequence {
private:
	std::shared_ptr<Sampler> sampler;
	long long instance;
public:
	RandomStateSequence(std::shared_ptr<Sampler> sampler, long long instance) :
		sampler(sampler), instance(instance) { }

	real sample() override {
		assert_info(sampler != nullptr, "null sampler");
		return sampler->sample(cursor++, instance);
	}
};

TC_NAMESPACE_END

