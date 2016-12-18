#include <taichi/visual/envmap.h>

TC_NAMESPACE_BEGIN
TC_INTERFACE_DEF(EnvironmentMap, "envmap");

TC_IMPLEMENTATION(EnvironmentMap, EnvironmentMap, "base");

void EnvironmentMap::initialize(const Config & config) {
	transform = Matrix4(1.0f);
	image = std::make_shared<ImageBuffer<Vector3>>(config.get_string("filepath"));
	width = image->get_width();
	height = image->get_height();
	/*
	for (int j = 0; j < height; j++) {
		// conversion
		auto scale = sin(pi * (0.5f + j) / height);
		for (int i = 0; i < width; i++) {
			(*image)[i][j] *= scale;
		}
	}
	*/
	for (int j = 0; j < height - j - 1; j++) {
		for (int i = 0; i < width; i++)
			std::swap((*image)[i][j], (*image)[i][height - j - 1]);
	}

	build_cdfs();
	/*
	P("test");
	P(uv_to_direction(Vector2(0.0f, 0.0f)));
	P(uv_to_direction(Vector2(0.0f, 0.5f)));
	P(uv_to_direction(Vector2(0.0f, 1.0f)));
	P(uv_to_direction(Vector2(0.5f, 0.25f)));
	P(uv_to_direction(Vector2(0.5f, 0.5f)));
	P(uv_to_direction(Vector2(0.5f, 0.75f)));
	P(uv_to_direction(Vector2(1.0f, 0.0f)));
	P(uv_to_direction(Vector2(1.0f, 0.5f)));
	P(uv_to_direction(Vector2(1.0f, 1.0f)));

	for (int i = 0; i < 10; i++) {
		real x = rand(), y = rand();
		auto uv = Vector2(x, y);
		P(uv);
		P(direction_to_uv(uv_to_direction(uv)));
	}

	for (int i = 0; i < 100; i++) {
		RandomStateSequence rand(create_instance<Sampler>("prand"), i);
		real pdf;
		Vector3 illum;
		Vector3 dir;
		dir = sample_direction(rand, pdf, illum);
		P(dir);
		P(pdf);
		P(illum);
		P(luminance(illum) / pdf);
	}
	*/
}

real EnvironmentMap::pdf(const Vector3 &dir) const {
	Vector2 uv = direction_to_uv(dir);
	return luminance(image->sample(uv.x, uv.y))
		/ avg_illum * (1.0f / 4 / pi);
}

Vector3 EnvironmentMap::sample_direction(StateSequence & rand, real & pdf, Vector3 & illum) const {
	Vector2 uv;
	real row_pdf, row_cdf;
	real col_pdf, col_cdf;
	real row_sample = rand();
	real col_sample = rand();
	int row = row_sampler.sample(row_sample, row_pdf, row_cdf);
	int col = col_samplers[row].sample(col_sample, col_pdf, col_cdf);
	real u = col + 0.5f;// (col_sample - col_cdf) / col_pdf;
	real v = row + 0.5f;// (row_sample - row_cdf) / row_pdf;
	uv.x = u / width;
	uv.y = v / height;
	illum = sample_illum(uv);
	pdf = row_pdf * col_pdf * width * height / sin(pi * (0.5f + row) / height);
	//P(luminance(illum) / pdf);
	return uv_to_direction(uv);
}

void EnvironmentMap::build_cdfs() {
	std::vector<real> row_pdf;
	avg_illum = 0;
	real total_weight = 0.0f;
	for (int j = 0; j < height; j++) {
		std::vector<real> col_pdf;
		real scale = sin(pi * (0.5f + j) / height);
		real total = 0.0f;
		for (int i = 0; i < width; i++) {
			real pdf = luminance(image->sample((i + 0.5f) / width, (j + 0.5f) / height));
			avg_illum += pdf * scale;
			total_weight += scale;
			total += pdf;
			col_pdf.push_back(pdf);
		}
		col_samplers.push_back(DiscreteSampler(col_pdf, true));
		row_pdf.push_back(total * scale);
	}
	row_sampler.initialize(row_pdf);
	avg_illum /= total_weight;
}

TC_NAMESPACE_END
