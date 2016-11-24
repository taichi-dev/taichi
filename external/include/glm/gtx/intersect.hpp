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
/// @ref gtx_intersect
/// @file glm/gtx/intersect.hpp
/// @date 2007-04-03 / 2011-06-07
/// @author Christophe Riccio
///
/// @see core (dependence)
/// @see gtx_closest_point (dependence)
///
/// @defgroup gtx_intersect GLM_GTX_intersect
/// @ingroup gtx
/// 
/// @brief Add intersection functions
/// 
/// <glm/gtx/intersect.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependency:
#include "../glm.hpp"
#include "../gtx/closest_point.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_closest_point extension included")
#endif

namespace glm
{
	/// @addtogroup gtx_intersect
	/// @{

	//! Compute the intersection of a ray and a triangle.
	//! Ray direction and plane normal must be unit length.
	//! From GLM_GTX_intersect extension.
	template <typename genType>
	GLM_FUNC_DECL bool intersectRayPlane(
		genType const & orig, genType const & dir,
		genType const & planeOrig, genType const & planeNormal,
		typename genType::value_type & intersectionDistance);

	//! Compute the intersection of a ray and a triangle.
	//! From GLM_GTX_intersect extension.
	template <typename genType>
	GLM_FUNC_DECL bool intersectRayTriangle(
		genType const & orig, genType const & dir,
		genType const & vert0, genType const & vert1, genType const & vert2,
		genType & baryPosition);

	//! Compute the intersection of a line and a triangle.
	//! From GLM_GTX_intersect extension.
	template <typename genType>
	GLM_FUNC_DECL bool intersectLineTriangle(
		genType const & orig, genType const & dir,
		genType const & vert0, genType const & vert1, genType const & vert2,
		genType & position);

	//! Compute the intersection distance of a ray and a sphere. 
	//! The ray direction vector is unit length.
	//! From GLM_GTX_intersect extension.
	template <typename genType>
	GLM_FUNC_DECL bool intersectRaySphere(
		genType const & rayStarting, genType const & rayNormalizedDirection,
		genType const & sphereCenter, typename genType::value_type const sphereRadiusSquered,
		typename genType::value_type & intersectionDistance);

	//! Compute the intersection of a ray and a sphere.
	//! From GLM_GTX_intersect extension.
	template <typename genType>
	GLM_FUNC_DECL bool intersectRaySphere(
		genType const & rayStarting, genType const & rayNormalizedDirection,
		genType const & sphereCenter, const typename genType::value_type sphereRadius,
		genType & intersectionPosition, genType & intersectionNormal);

	//! Compute the intersection of a line and a sphere.
	//! From GLM_GTX_intersect extension
	template <typename genType>
	GLM_FUNC_DECL bool intersectLineSphere(
		genType const & point0, genType const & point1,
		genType const & sphereCenter, typename genType::value_type sphereRadius,
		genType & intersectionPosition1, genType & intersectionNormal1, 
		genType & intersectionPosition2 = genType(), genType & intersectionNormal2 = genType());

	/// @}
}//namespace glm

#include "intersect.inl"
