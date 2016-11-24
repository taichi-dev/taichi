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
/// @ref gtx_associated_min_max
/// @file glm/gtx/associated_min_max.hpp
/// @date 2008-03-10 / 2014-10-11
/// @author Christophe Riccio
///
/// @see core (dependence)
/// @see gtx_extented_min_max (dependence)
///
/// @defgroup gtx_associated_min_max GLM_GTX_associated_min_max
/// @ingroup gtx
/// 
/// @brief Min and max functions that return associated values not the compared onces.
/// <glm/gtx/associated_min_max.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependency:
#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_associated_min_max extension included")
#endif

namespace glm
{
	/// @addtogroup gtx_associated_min_max
	/// @{

	/// Minimum comparison between 2 variables and returns 2 associated variable values
	/// @see gtx_associated_min_max
	template<typename T, typename U, precision P>
	GLM_FUNC_DECL U associatedMin(T x, U a, T y, U b);

	/// Minimum comparison between 2 variables and returns 2 associated variable values
	/// @see gtx_associated_min_max
	template<typename T, typename U, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL tvec2<U, P> associatedMin(
		vecType<T, P> const & x, vecType<U, P> const & a,
		vecType<T, P> const & y, vecType<U, P> const & b);

	/// Minimum comparison between 2 variables and returns 2 associated variable values
	/// @see gtx_associated_min_max
	template<typename T, typename U, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<U, P> associatedMin(
		T x, const vecType<U, P>& a,
		T y, const vecType<U, P>& b);

	/// Minimum comparison between 2 variables and returns 2 associated variable values
	/// @see gtx_associated_min_max
	template<typename T, typename U, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<U, P> associatedMin(
		vecType<T, P> const & x, U a,
		vecType<T, P> const & y, U b);

	/// Minimum comparison between 3 variables and returns 3 associated variable values
	/// @see gtx_associated_min_max
	template<typename T, typename U>
	GLM_FUNC_DECL U associatedMin(
		T x, U a,
		T y, U b,
		T z, U c);

	/// Minimum comparison between 3 variables and returns 3 associated variable values
	/// @see gtx_associated_min_max
	template<typename T, typename U, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<U, P> associatedMin(
		vecType<T, P> const & x, vecType<U, P> const & a,
		vecType<T, P> const & y, vecType<U, P> const & b,
		vecType<T, P> const & z, vecType<U, P> const & c);

	/// Minimum comparison between 4 variables and returns 4 associated variable values
	/// @see gtx_associated_min_max
	template<typename T, typename U>
	GLM_FUNC_DECL U associatedMin(
		T x, U a,
		T y, U b,
		T z, U c,
		T w, U d);

	/// Minimum comparison between 4 variables and returns 4 associated variable values
	/// @see gtx_associated_min_max
	template<typename T, typename U, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<U, P> associatedMin(
		vecType<T, P> const & x, vecType<U, P> const & a,
		vecType<T, P> const & y, vecType<U, P> const & b,
		vecType<T, P> const & z, vecType<U, P> const & c,
		vecType<T, P> const & w, vecType<U, P> const & d);

	/// Minimum comparison between 4 variables and returns 4 associated variable values
	/// @see gtx_associated_min_max
	template<typename T, typename U, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<U, P> associatedMin(
		T x, vecType<U, P> const & a,
		T y, vecType<U, P> const & b,
		T z, vecType<U, P> const & c,
		T w, vecType<U, P> const & d);

	/// Minimum comparison between 4 variables and returns 4 associated variable values
	/// @see gtx_associated_min_max
	template<typename T, typename U, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<U, P> associatedMin(
		vecType<T, P> const & x, U a,
		vecType<T, P> const & y, U b,
		vecType<T, P> const & z, U c,
		vecType<T, P> const & w, U d);

	/// Maximum comparison between 2 variables and returns 2 associated variable values
	/// @see gtx_associated_min_max
	template<typename T, typename U>
	GLM_FUNC_DECL U associatedMax(T x, U a, T y, U b);

	/// Maximum comparison between 2 variables and returns 2 associated variable values
	/// @see gtx_associated_min_max
	template<typename T, typename U, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL tvec2<U, P> associatedMax(
		vecType<T, P> const & x, vecType<U, P> const & a,
		vecType<T, P> const & y, vecType<U, P> const & b);

	/// Maximum comparison between 2 variables and returns 2 associated variable values
	/// @see gtx_associated_min_max
	template<typename T, typename U, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> associatedMax(
		T x, vecType<U, P> const & a,
		T y, vecType<U, P> const & b);

	/// Maximum comparison between 2 variables and returns 2 associated variable values
	/// @see gtx_associated_min_max
	template<typename T, typename U, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<U, P> associatedMax(
		vecType<T, P> const & x, U a,
		vecType<T, P> const & y, U b);

	/// Maximum comparison between 3 variables and returns 3 associated variable values
	/// @see gtx_associated_min_max
	template<typename T, typename U>
	GLM_FUNC_DECL U associatedMax(
		T x, U a,
		T y, U b,
		T z, U c);

	/// Maximum comparison between 3 variables and returns 3 associated variable values
	/// @see gtx_associated_min_max
	template<typename T, typename U, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<U, P> associatedMax(
		vecType<T, P> const & x, vecType<U, P> const & a,
		vecType<T, P> const & y, vecType<U, P> const & b,
		vecType<T, P> const & z, vecType<U, P> const & c);

	/// Maximum comparison between 3 variables and returns 3 associated variable values
	/// @see gtx_associated_min_max
	template<typename T, typename U, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> associatedMax(
		T x, vecType<U, P> const & a,
		T y, vecType<U, P> const & b,
		T z, vecType<U, P> const & c);

	/// Maximum comparison between 3 variables and returns 3 associated variable values
	/// @see gtx_associated_min_max
	template<typename T, typename U, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<U, P> associatedMax(
		vecType<T, P> const & x, U a,
		vecType<T, P> const & y, U b,
		vecType<T, P> const & z, U c);

	/// Maximum comparison between 4 variables and returns 4 associated variable values
	/// @see gtx_associated_min_max
	template<typename T, typename U>
	GLM_FUNC_DECL U associatedMax(
		T x, U a,
		T y, U b,
		T z, U c,
		T w, U d);

	/// Maximum comparison between 4 variables and returns 4 associated variable values
	/// @see gtx_associated_min_max
	template<typename T, typename U, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<U, P> associatedMax(
		vecType<T, P> const & x, vecType<U, P> const & a,
		vecType<T, P> const & y, vecType<U, P> const & b,
		vecType<T, P> const & z, vecType<U, P> const & c,
		vecType<T, P> const & w, vecType<U, P> const & d);

	/// Maximum comparison between 4 variables and returns 4 associated variable values
	/// @see gtx_associated_min_max
	template<typename T, typename U, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<U, P> associatedMax(
		T x, vecType<U, P> const & a,
		T y, vecType<U, P> const & b,
		T z, vecType<U, P> const & c,
		T w, vecType<U, P> const & d);

	/// Maximum comparison between 4 variables and returns 4 associated variable values
	/// @see gtx_associated_min_max
	template<typename T, typename U, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<U, P> associatedMax(
		vecType<T, P> const & x, U a,
		vecType<T, P> const & y, U b,
		vecType<T, P> const & z, U c,
		vecType<T, P> const & w, U d);

	/// @}
} //namespace glm

#include "associated_min_max.inl"
