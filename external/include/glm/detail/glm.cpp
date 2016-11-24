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
/// @ref core
/// @file glm/glm.cpp
/// @date 2013-04-22 / 2013-04-22
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/dual_quaternion.hpp>

namespace glm
{
// tvec1 type explicit instantiation
template struct tvec1<uint8, lowp>;
template struct tvec1<uint16, lowp>;
template struct tvec1<uint32, lowp>;
template struct tvec1<uint64, lowp>;
template struct tvec1<int8, lowp>;
template struct tvec1<int16, lowp>;
template struct tvec1<int32, lowp>;
template struct tvec1<int64, lowp>;
template struct tvec1<float32, lowp>;
template struct tvec1<float64, lowp>;

template struct tvec1<uint8, mediump>;
template struct tvec1<uint16, mediump>;
template struct tvec1<uint32, mediump>;
template struct tvec1<uint64, mediump>;
template struct tvec1<int8, mediump>;
template struct tvec1<int16, mediump>;
template struct tvec1<int32, mediump>;
template struct tvec1<int64, mediump>;
template struct tvec1<float32, mediump>;
template struct tvec1<float64, mediump>;

template struct tvec1<uint8, highp>;
template struct tvec1<uint16, highp>;
template struct tvec1<uint32, highp>;
template struct tvec1<uint64, highp>;
template struct tvec1<int8, highp>;
template struct tvec1<int16, highp>;
template struct tvec1<int32, highp>;
template struct tvec1<int64, highp>;
template struct tvec1<float32, highp>;
template struct tvec1<float64, highp>;

// tvec2 type explicit instantiation
template struct tvec2<uint8, lowp>;
template struct tvec2<uint16, lowp>;
template struct tvec2<uint32, lowp>;
template struct tvec2<uint64, lowp>;
template struct tvec2<int8, lowp>;
template struct tvec2<int16, lowp>;
template struct tvec2<int32, lowp>;
template struct tvec2<int64, lowp>;
template struct tvec2<float32, lowp>;
template struct tvec2<float64, lowp>;

template struct tvec2<uint8, mediump>;
template struct tvec2<uint16, mediump>;
template struct tvec2<uint32, mediump>;
template struct tvec2<uint64, mediump>;
template struct tvec2<int8, mediump>;
template struct tvec2<int16, mediump>;
template struct tvec2<int32, mediump>;
template struct tvec2<int64, mediump>;
template struct tvec2<float32, mediump>;
template struct tvec2<float64, mediump>;

template struct tvec2<uint8, highp>;
template struct tvec2<uint16, highp>;
template struct tvec2<uint32, highp>;
template struct tvec2<uint64, highp>;
template struct tvec2<int8, highp>;
template struct tvec2<int16, highp>;
template struct tvec2<int32, highp>;
template struct tvec2<int64, highp>;
template struct tvec2<float32, highp>;
template struct tvec2<float64, highp>;

// tvec3 type explicit instantiation
template struct tvec3<uint8, lowp>;
template struct tvec3<uint16, lowp>;
template struct tvec3<uint32, lowp>;
template struct tvec3<uint64, lowp>;
template struct tvec3<int8, lowp>;
template struct tvec3<int16, lowp>;
template struct tvec3<int32, lowp>;
template struct tvec3<int64, lowp>;
template struct tvec3<float32, lowp>;
template struct tvec3<float64, lowp>;

template struct tvec3<uint8, mediump>;
template struct tvec3<uint16, mediump>;
template struct tvec3<uint32, mediump>;
template struct tvec3<uint64, mediump>;
template struct tvec3<int8, mediump>;
template struct tvec3<int16, mediump>;
template struct tvec3<int32, mediump>;
template struct tvec3<int64, mediump>;
template struct tvec3<float32, mediump>;
template struct tvec3<float64, mediump>;

template struct tvec3<uint8, highp>;
template struct tvec3<uint16, highp>;
template struct tvec3<uint32, highp>;
template struct tvec3<uint64, highp>;
template struct tvec3<int8, highp>;
template struct tvec3<int16, highp>;
template struct tvec3<int32, highp>;
template struct tvec3<int64, highp>;
template struct tvec3<float32, highp>;
template struct tvec3<float64, highp>;

// tvec4 type explicit instantiation
template struct tvec4<uint8, lowp>;
template struct tvec4<uint16, lowp>;
template struct tvec4<uint32, lowp>;
template struct tvec4<uint64, lowp>;
template struct tvec4<int8, lowp>;
template struct tvec4<int16, lowp>;
template struct tvec4<int32, lowp>;
template struct tvec4<int64, lowp>;
template struct tvec4<float32, lowp>;
template struct tvec4<float64, lowp>;

template struct tvec4<uint8, mediump>;
template struct tvec4<uint16, mediump>;
template struct tvec4<uint32, mediump>;
template struct tvec4<uint64, mediump>;
template struct tvec4<int8, mediump>;
template struct tvec4<int16, mediump>;
template struct tvec4<int32, mediump>;
template struct tvec4<int64, mediump>;
template struct tvec4<float32, mediump>;
template struct tvec4<float64, mediump>;

template struct tvec4<uint8, highp>;
template struct tvec4<uint16, highp>;
template struct tvec4<uint32, highp>;
template struct tvec4<uint64, highp>;
template struct tvec4<int8, highp>;
template struct tvec4<int16, highp>;
template struct tvec4<int32, highp>;
template struct tvec4<int64, highp>;
template struct tvec4<float32, highp>;
template struct tvec4<float64, highp>;

// tmat2x2 type explicit instantiation
template struct tmat2x2<float32, lowp>;
template struct tmat2x2<float64, lowp>;

template struct tmat2x2<float32, mediump>;
template struct tmat2x2<float64, mediump>;

template struct tmat2x2<float32, highp>;
template struct tmat2x2<float64, highp>;

// tmat2x3 type explicit instantiation
template struct tmat2x3<float32, lowp>;
template struct tmat2x3<float64, lowp>;

template struct tmat2x3<float32, mediump>;
template struct tmat2x3<float64, mediump>;

template struct tmat2x3<float32, highp>;
template struct tmat2x3<float64, highp>;

// tmat2x4 type explicit instantiation
template struct tmat2x4<float32, lowp>;
template struct tmat2x4<float64, lowp>;

template struct tmat2x4<float32, mediump>;
template struct tmat2x4<float64, mediump>;

template struct tmat2x4<float32, highp>;
template struct tmat2x4<float64, highp>;

// tmat3x2 type explicit instantiation
template struct tmat3x2<float32, lowp>;
template struct tmat3x2<float64, lowp>;

template struct tmat3x2<float32, mediump>;
template struct tmat3x2<float64, mediump>;

template struct tmat3x2<float32, highp>;
template struct tmat3x2<float64, highp>;

// tmat3x3 type explicit instantiation
template struct tmat3x3<float32, lowp>;
template struct tmat3x3<float64, lowp>;

template struct tmat3x3<float32, mediump>;
template struct tmat3x3<float64, mediump>;

template struct tmat3x3<float32, highp>;
template struct tmat3x3<float64, highp>;

// tmat3x4 type explicit instantiation
template struct tmat3x4<float32, lowp>;
template struct tmat3x4<float64, lowp>;

template struct tmat3x4<float32, mediump>;
template struct tmat3x4<float64, mediump>;

template struct tmat3x4<float32, highp>;
template struct tmat3x4<float64, highp>;

// tmat4x2 type explicit instantiation
template struct tmat4x2<float32, lowp>;
template struct tmat4x2<float64, lowp>;

template struct tmat4x2<float32, mediump>;
template struct tmat4x2<float64, mediump>;

template struct tmat4x2<float32, highp>;
template struct tmat4x2<float64, highp>;

// tmat4x3 type explicit instantiation
template struct tmat4x3<float32, lowp>;
template struct tmat4x3<float64, lowp>;

template struct tmat4x3<float32, mediump>;
template struct tmat4x3<float64, mediump>;

template struct tmat4x3<float32, highp>;
template struct tmat4x3<float64, highp>;

// tmat4x4 type explicit instantiation
template struct tmat4x4<float32, lowp>;
template struct tmat4x4<float64, lowp>;

template struct tmat4x4<float32, mediump>;
template struct tmat4x4<float64, mediump>;

template struct tmat4x4<float32, highp>;
template struct tmat4x4<float64, highp>;

// tquat type explicit instantiation
template struct tquat<float32, lowp>;
template struct tquat<float64, lowp>;

template struct tquat<float32, mediump>;
template struct tquat<float64, mediump>;

template struct tquat<float32, highp>;
template struct tquat<float64, highp>;

//tdualquat type explicit instantiation
template struct tdualquat<float32, lowp>;
template struct tdualquat<float64, lowp>;

template struct tdualquat<float32, mediump>;
template struct tdualquat<float64, mediump>;

template struct tdualquat<float32, highp>;
template struct tdualquat<float64, highp>;

}//namespace glm

