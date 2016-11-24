///////////////////////////////////////////////////////////////////////////////////
/// OpenGL Mathematics (glm.g-truc.net)
///
/// Copyright (c) 2005 - 2015 G-Truc Creation (www.g-truc.net)
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to deal
/// in the Software without restriction, including without limitation the rights
/// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
/// copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
/// 
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
/// 
/// Restrictions:
///		By making use of the Software for military purposes, you choose to make
///		a Bunny unhappy.
/// 
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
/// THE SOFTWARE.
///
/// @file glm/glm.hpp
/// @date 2009-05-01 / 2011-05-16
/// @author Christophe Riccio
///
/// @ref core (Dependence)
/// 
/// @defgroup gtc GTC Extensions (Stable)
///
/// @brief Functions and types that the GLSL specification doesn't define, but useful to have for a C++ program.
/// 
/// GTC extensions aim to be stable. 
/// 
/// Even if it's highly unrecommended, it's possible to include all the extensions at once by
/// including <glm/ext.hpp>. Otherwise, each extension needs to be included  a specific file.
/// 
/// @defgroup gtx GTX Extensions (Experimental)
/// 
/// @brief Functions and types that the GLSL specification doesn't define, but 
/// useful to have for a C++ program.
/// 
/// Experimental extensions are useful functions and types, but the development of
/// their API and functionality is not necessarily stable. They can change 
/// substantially between versions. Backwards compatibility is not much of an issue
/// for them.
/// 
/// Even if it's highly unrecommended, it's possible to include all the extensions 
/// at once by including <glm/ext.hpp>. Otherwise, each extension needs to be 
/// included  a specific file.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

#if(defined(GLM_MESSAGES) && !defined(GLM_MESSAGE_EXT_INCLUDED_DISPLAYED))
#	define GLM_MESSAGE_EXT_INCLUDED_DISPLAYED
#	pragma message("GLM: All extensions included (not recommanded)")
#endif//GLM_MESSAGES

#include "./gtc/bitfield.hpp"
#include "./gtc/constants.hpp"
#include "./gtc/epsilon.hpp"
#include "./gtc/integer.hpp"
#include "./gtc/matrix_access.hpp"
#include "./gtc/matrix_integer.hpp"
#include "./gtc/matrix_inverse.hpp"
#include "./gtc/matrix_transform.hpp"
#include "./gtc/noise.hpp"
#include "./gtc/packing.hpp"
#include "./gtc/quaternion.hpp"
#include "./gtc/random.hpp"
#include "./gtc/reciprocal.hpp"
#include "./gtc/round.hpp"
#include "./gtc/type_precision.hpp"
#include "./gtc/type_ptr.hpp"
#include "./gtc/ulp.hpp"
#include "./gtc/vec1.hpp"

#include "./gtx/associated_min_max.hpp"
#include "./gtx/bit.hpp"
#include "./gtx/closest_point.hpp"
#include "./gtx/color_space.hpp"
#include "./gtx/color_space_YCoCg.hpp"
#include "./gtx/compatibility.hpp"
#include "./gtx/component_wise.hpp"
#include "./gtx/dual_quaternion.hpp"
#include "./gtx/euler_angles.hpp"
#include "./gtx/extend.hpp"
#include "./gtx/extented_min_max.hpp"
#include "./gtx/fast_exponential.hpp"
#include "./gtx/fast_square_root.hpp"
#include "./gtx/fast_trigonometry.hpp"
#include "./gtx/gradient_paint.hpp"
#include "./gtx/handed_coordinate_space.hpp"
#include "./gtx/integer.hpp"
#include "./gtx/intersect.hpp"
#include "./gtx/log_base.hpp"
#include "./gtx/matrix_cross_product.hpp"
#include "./gtx/matrix_interpolation.hpp"
#include "./gtx/matrix_major_storage.hpp"
#include "./gtx/matrix_operation.hpp"
#include "./gtx/matrix_query.hpp"
#include "./gtx/mixed_product.hpp"
#include "./gtx/norm.hpp"
#include "./gtx/normal.hpp"
#include "./gtx/normalize_dot.hpp"
#include "./gtx/number_precision.hpp"
#include "./gtx/optimum_pow.hpp"
#include "./gtx/orthonormalize.hpp"
#include "./gtx/perpendicular.hpp"
#include "./gtx/polar_coordinates.hpp"
#include "./gtx/projection.hpp"
#include "./gtx/quaternion.hpp"
#include "./gtx/raw_data.hpp"
#include "./gtx/rotate_vector.hpp"
#include "./gtx/spline.hpp"
#include "./gtx/std_based_type.hpp"
#if !(GLM_COMPILER & GLM_COMPILER_CUDA)
#	include "./gtx/string_cast.hpp"
#endif
#include "./gtx/transform.hpp"
#include "./gtx/transform2.hpp"
#include "./gtx/type_aligned.hpp"
#include "./gtx/vector_angle.hpp"
#include "./gtx/vector_query.hpp"
#include "./gtx/wrap.hpp"

#if GLM_HAS_TEMPLATE_ALIASES
#	include "./gtx/scalar_multiplication.hpp"
#endif

#if GLM_HAS_RANGE_FOR
#	include "./gtx/range.hpp"
#endif

#if GLM_ARCH & GLM_ARCH_SSE2
#	include "./gtx/simd_vec4.hpp"
#	include "./gtx/simd_mat4.hpp"
#endif
