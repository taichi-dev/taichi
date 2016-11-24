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
/// @ref gtx_number_precision
/// @file glm/gtx/number_precision.hpp
/// @date 2007-05-10 / 2011-06-07
/// @author Christophe Riccio
///
/// @see core (dependence)
/// @see gtc_type_precision (dependence)
/// @see gtc_quaternion (dependence)
///
/// @defgroup gtx_number_precision GLM_GTX_number_precision
/// @ingroup gtx
/// 
/// @brief Defined size types.
/// 
/// <glm/gtx/number_precision.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependency:
#include "../glm.hpp"
#include "../gtc/type_precision.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_number_precision extension included")
#endif

namespace glm{
namespace gtx
{
	/////////////////////////////
	// Unsigned int vector types 

	/// @addtogroup gtx_number_precision
	/// @{

	typedef u8			u8vec1;		//!< \brief 8bit unsigned integer scalar. (from GLM_GTX_number_precision extension)
	typedef u16			u16vec1;    //!< \brief 16bit unsigned integer scalar. (from GLM_GTX_number_precision extension)
	typedef u32			u32vec1;    //!< \brief 32bit unsigned integer scalar. (from GLM_GTX_number_precision extension)
	typedef u64			u64vec1;    //!< \brief 64bit unsigned integer scalar. (from GLM_GTX_number_precision extension)

	//////////////////////
	// Float vector types 

	typedef f32			f32vec1;    //!< \brief Single-precision floating-point scalar. (from GLM_GTX_number_precision extension)
	typedef f64			f64vec1;    //!< \brief Single-precision floating-point scalar. (from GLM_GTX_number_precision extension)

	//////////////////////
	// Float matrix types 

	typedef f32			f32mat1;	//!< \brief Single-precision floating-point scalar. (from GLM_GTX_number_precision extension)
	typedef f32			f32mat1x1;	//!< \brief Single-precision floating-point scalar. (from GLM_GTX_number_precision extension)
	typedef f64			f64mat1;	//!< \brief Double-precision floating-point scalar. (from GLM_GTX_number_precision extension)
	typedef f64			f64mat1x1;	//!< \brief Double-precision floating-point scalar. (from GLM_GTX_number_precision extension)

	/// @}
}//namespace gtx
}//namespace glm

#include "number_precision.inl"
