#pragma once

#include <taichi/math/linalg.h>
#include <vector>
#include <algorithm>
#include <numeric>

TC_NAMESPACE_BEGIN

class DiscreteSampler {
private:
	std::vector<real> pdf;
	std::vector<real> cdf;
public:
	DiscreteSampler() {}
	DiscreteSampler(std::vector<real> &pdf, bool allow_zero_total_pdf = false) {
		assert_info(pdf.size() > 0, "No choice for sampler.");
		initialize(pdf, allow_zero_total_pdf);
	}
	int get_num_elements() const {
		return (int)pdf.size();
	}
	void initialize(std::vector<real> unnormalized_pdf, bool allow_zero_total_pdf = false) {
		//float sum = std::accumulate(unnormalized_pdf.begin(), unnormalized_pdf.end(), 0.0f);
		float sum = 0.0f;
		for (int i = 0; i < (int)unnormalized_pdf.size(); i++) {
			real pdf = unnormalized_pdf[i];
			assert_info(pdf >= 0, "No negative pdf allowed!");
			sum += pdf;
		}
		if (!allow_zero_total_pdf) {
			assert_info(sum > 0, "Sum of pdf is zero.");
		}
		else {
			if (sum == 0)
				return;
		}
		float inv_sum = 1.0f / sum;
		this->pdf.resize(unnormalized_pdf.size());
		this->cdf.resize(unnormalized_pdf.size());
		for (int i = 0; i < (int)pdf.size(); i++) {
			pdf[i] = unnormalized_pdf[i] * inv_sum;
			cdf[i] = pdf[i];
			if (i > 0) {
				cdf[i] += cdf[i - 1];
			}
		}
	}
	int sample(real r, real &pdf_out) const {
		int index = int(std::lower_bound(cdf.begin(), cdf.end(), r) - cdf.begin());
		index = std::min(index, get_num_elements() - 1);
		pdf_out = pdf[index];
		return index;
	}
	int sample(real r, real &pdf_out, real &cdf_out) const {
		int index = int(std::lower_bound(cdf.begin(), cdf.end(), r) - cdf.begin());
		index = std::min(index, get_num_elements() - 1);
		pdf_out = pdf[index];
		cdf_out = cdf[index];
		return index;
	}
	int sample(real r) const {
		real _;
		return sample(r, _);
	}
	real get_pdf(int id) const {
		return pdf[id];
	}
	real get_cdf(int id) const {
		return cdf[id];
	}
};

inline void test_discrete_sampler() {
	const int n = 10;
	std::vector<real> p;
	for (int i = 0; i < n; i++) {
		p.push_back((real)i);
	}
	DiscreteSampler ds(p);
	int count[n];
	for (int i = 0; i < 1000000 * (n - 1) * (n) / 2; i++) {
		count[ds.sample(rand())]++;
	}
	for (int i = 0; i < n; i++) {
		P(i);
		P(count[i]);
	}
}

TC_NAMESPACE_END

