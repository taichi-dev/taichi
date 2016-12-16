#include "python_interface.h"
#include <taichi/math/array_2d.h>

TC_NAMESPACE_BEGIN
template<typename T>
void array2d_to_ndarray(T *arr, long long output) // actually as pointer...
{
	int width = arr->get_width(), height = arr->get_height();
	for (auto &ind : arr->get_region()) {
		reinterpret_cast<float *>(output)[ind.i + ind.j * width] = (*arr)[ind];
	}
}

template<typename T>
void image_buffer_to_ndarray(T *arr, long long output) // actually as pointer...
{
	int width = arr->get_width(), height = arr->get_height();
	for (auto &ind : arr->get_region()) {
		for (int i = 0; i < 3; i++) {
			reinterpret_cast<float *>(output)[ind.i * 3 + ind.j * width * 3 + i] = (*arr)[ind][i];
		}
	}
}

template void array2d_to_ndarray(Array2D<float> *arr, long long);
template void image_buffer_to_ndarray(ImageBuffer<Vector3> *arr, long long);
TC_NAMESPACE_END

