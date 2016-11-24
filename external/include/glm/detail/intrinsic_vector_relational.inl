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
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
/// THE SOFTWARE.
///
/// @ref core
/// @file glm/detail/intrinsic_vector_relational.inl
/// @date 2009-06-09 / 2011-06-15
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////
//
//// lessThan
//template <typename valType>
//GLM_FUNC_QUALIFIER typename tvec2<T, P>::bool_type lessThan
//(
//	tvec2<T, P> const & x, 
//	tvec2<T, P> const & y
//)
//{
//	GLM_STATIC_ASSERT(
//		detail::type<valType>::is_float || 
//		detail::type<valType>::is_int || 
//		detail::type<valType>::is_uint);
//
//    return typename tvec2<bool>::bool_type(x.x < y.x, x.y < y.y);
//}
//
//template <typename valType>
//GLM_FUNC_QUALIFIER typename tvec3<T, P>::bool_type lessThan
//(
//	tvec3<T, P> const & x, 
//	tvec3<T, P> const & y
//)
//{
//	GLM_STATIC_ASSERT(
//		detail::type<valType>::is_float || 
//		detail::type<valType>::is_int || 
//		detail::type<valType>::is_uint);
//	
//	return typename tvec3<bool>::bool_type(x.x < y.x, x.y < y.y, x.z < y.z);
//}
//
//template <typename valType>
//GLM_FUNC_QUALIFIER typename tvec4<T, P>::bool_type lessThan
//(
//	tvec4<T, P> const & x, 
//	tvec4<T, P> const & y
//)
//{
//	GLM_STATIC_ASSERT(
//		detail::type<valType>::is_float || 
//		detail::type<valType>::is_int || 
//		detail::type<valType>::is_uint);
//
//	return typename tvec4<bool>::bool_type(x.x < y.x, x.y < y.y, x.z < y.z, x.w < y.w);
//}
//
//// lessThanEqual
//template <typename valType>
//GLM_FUNC_QUALIFIER typename tvec2<T, P>::bool_type lessThanEqual
//(
//	tvec2<T, P> const & x, 
//	tvec2<T, P> const & y
//)
//{
//	GLM_STATIC_ASSERT(
//		detail::type<valType>::is_float || 
//		detail::type<valType>::is_int || 
//		detail::type<valType>::is_uint);
//
//	return typename tvec2<bool>::bool_type(x.x <= y.x, x.y <= y.y);
//}
//
//template <typename valType>
//GLM_FUNC_QUALIFIER typename tvec3<T, P>::bool_type lessThanEqual
//(
//	tvec3<T, P> const & x, 
//	tvec3<T, P> const & y
//)
//{
//	GLM_STATIC_ASSERT(
//		detail::type<valType>::is_float || 
//		detail::type<valType>::is_int || 
//		detail::type<valType>::is_uint);
//	
//	return typename tvec3<bool>::bool_type(x.x <= y.x, x.y <= y.y, x.z <= y.z);
//}
//
//template <typename valType>
//GLM_FUNC_QUALIFIER typename tvec4<T, P>::bool_type lessThanEqual
//(
//	tvec4<T, P> const & x, 
//	tvec4<T, P> const & y
//)
//{
//	GLM_STATIC_ASSERT(
//		detail::type<valType>::is_float || 
//		detail::type<valType>::is_int || 
//		detail::type<valType>::is_uint);
//	
//	return typename tvec4<bool>::bool_type(x.x <= y.x, x.y <= y.y, x.z <= y.z, x.w <= y.w);
//}
//
//// greaterThan
//template <typename valType>
//GLM_FUNC_QUALIFIER typename tvec2<T, P>::bool_type greaterThan
//(
//	tvec2<T, P> const & x, 
//	tvec2<T, P> const & y
//)
//{
//	GLM_STATIC_ASSERT(
//		detail::type<valType>::is_float || 
//		detail::type<valType>::is_int || 
//		detail::type<valType>::is_uint);
//
//	return typename tvec2<bool>::bool_type(x.x > y.x, x.y > y.y);
//}
//
//template <typename valType>
//GLM_FUNC_QUALIFIER typename tvec3<T, P>::bool_type greaterThan
//(
//	tvec3<T, P> const & x, 
//	tvec3<T, P> const & y
//)
//{
//	GLM_STATIC_ASSERT(
//		detail::type<valType>::is_float || 
//		detail::type<valType>::is_int || 
//		detail::type<valType>::is_uint);
//	
//	return typename tvec3<bool>::bool_type(x.x > y.x, x.y > y.y, x.z > y.z);
//}
//
//template <typename valType>
//GLM_FUNC_QUALIFIER typename tvec4<T, P>::bool_type greaterThan
//(
//	tvec4<T, P> const & x, 
//	tvec4<T, P> const & y
//)
//{
//	GLM_STATIC_ASSERT(
//		detail::type<valType>::is_float || 
//		detail::type<valType>::is_int || 
//		detail::type<valType>::is_uint);
//	
//	return typename tvec4<bool>::bool_type(x.x > y.x, x.y > y.y, x.z > y.z, x.w > y.w);
//}
//
//// greaterThanEqual
//template <typename valType>
//GLM_FUNC_QUALIFIER typename tvec2<T, P>::bool_type greaterThanEqual
//(
//	tvec2<T, P> const & x, 
//	tvec2<T, P> const & y
//)
//{
//	GLM_STATIC_ASSERT(
//		detail::type<valType>::is_float || 
//		detail::type<valType>::is_int || 
//		detail::type<valType>::is_uint);
//
//	return typename tvec2<bool>::bool_type(x.x >= y.x, x.y >= y.y);
//}
//
//template <typename valType>
//GLM_FUNC_QUALIFIER typename tvec3<T, P>::bool_type greaterThanEqual
//(
//	tvec3<T, P> const & x, 
//	tvec3<T, P> const & y
//)
//{
//	GLM_STATIC_ASSERT(
//		detail::type<valType>::is_float || 
//		detail::type<valType>::is_int || 
//		detail::type<valType>::is_uint);
//
//	return typename tvec3<bool>::bool_type(x.x >= y.x, x.y >= y.y, x.z >= y.z);
//}
//
//template <typename valType>
//GLM_FUNC_QUALIFIER typename tvec4<T, P>::bool_type greaterThanEqual
//(
//	tvec4<T, P> const & x, 
//	tvec4<T, P> const & y
//)
//{
//	GLM_STATIC_ASSERT(
//		detail::type<valType>::is_float || 
//		detail::type<valType>::is_int || 
//		detail::type<valType>::is_uint);
//
//	return typename tvec4<bool>::bool_type(x.x >= y.x, x.y >= y.y, x.z >= y.z, x.w >= y.w);
//}
//
//// equal
//template <typename valType>
//GLM_FUNC_QUALIFIER typename tvec2<T, P>::bool_type equal
//(
//	tvec2<T, P> const & x, 
//	tvec2<T, P> const & y
//)
//{
//	GLM_STATIC_ASSERT(
//		detail::type<valType>::is_float || 
//		detail::type<valType>::is_int || 
//		detail::type<valType>::is_uint || 
//		detail::type<valType>::is_bool);
//
//	return typename tvec2<T, P>::bool_type(x.x == y.x, x.y == y.y);
//}
//
//template <typename valType>
//GLM_FUNC_QUALIFIER typename tvec3<T, P>::bool_type equal
//(
//	tvec3<T, P> const & x, 
//	tvec3<T, P> const & y
//)
//{
//	GLM_STATIC_ASSERT(
//		detail::type<valType>::is_float || 
//		detail::type<valType>::is_int || 
//		detail::type<valType>::is_uint || 
//		detail::type<valType>::is_bool);
//
//	return typename tvec3<T, P>::bool_type(x.x == y.x, x.y == y.y, x.z == y.z);
//}
//
//template <typename valType>
//GLM_FUNC_QUALIFIER typename tvec4<T, P>::bool_type equal
//(
//	tvec4<T, P> const & x, 
//	tvec4<T, P> const & y
//)
//{
//	GLM_STATIC_ASSERT(
//		detail::type<valType>::is_float || 
//		detail::type<valType>::is_int || 
//		detail::type<valType>::is_uint || 
//		detail::type<valType>::is_bool);
//
//	return typename tvec4<T, P>::bool_type(x.x == y.x, x.y == y.y, x.z == y.z, x.w == y.w);
//}
//
//// notEqual
//template <typename valType>
//GLM_FUNC_QUALIFIER typename tvec2<T, P>::bool_type notEqual
//(
//	tvec2<T, P> const & x, 
//	tvec2<T, P> const & y
//)
//{
//	GLM_STATIC_ASSERT(
//		detail::type<valType>::is_float || 
//		detail::type<valType>::is_int || 
//		detail::type<valType>::is_uint || 
//		detail::type<valType>::is_bool);
//
//	return typename tvec2<T, P>::bool_type(x.x != y.x, x.y != y.y);
//}
//
//template <typename valType>
//GLM_FUNC_QUALIFIER typename tvec3<T, P>::bool_type notEqual
//(
//	tvec3<T, P> const & x, 
//	tvec3<T, P> const & y
//)
//{
//	GLM_STATIC_ASSERT(
//		detail::type<valType>::is_float || 
//		detail::type<valType>::is_int || 
//		detail::type<valType>::is_uint || 
//		detail::type<valType>::is_bool);
//
//	return typename tvec3<T, P>::bool_type(x.x != y.x, x.y != y.y, x.z != y.z);
//}
//
//template <typename valType>
//GLM_FUNC_QUALIFIER typename tvec4<T, P>::bool_type notEqual
//(
//	tvec4<T, P> const & x, 
//	tvec4<T, P> const & y
//)
//{
//	GLM_STATIC_ASSERT(
//		detail::type<valType>::is_float || 
//		detail::type<valType>::is_int || 
//		detail::type<valType>::is_uint || 
//		detail::type<valType>::is_bool);
//
//	return typename tvec4<T, P>::bool_type(x.x != y.x, x.y != y.y, x.z != y.z, x.w != y.w);
//}
//
//// any
//GLM_FUNC_QUALIFIER bool any(tvec2<bool> const & x)
//{
//	return x.x || x.y;
//}
//
//GLM_FUNC_QUALIFIER bool any(tvec3<bool> const & x)
//{
//    return x.x || x.y || x.z;
//}
//
//GLM_FUNC_QUALIFIER bool any(tvec4<bool> const & x)
//{
//    return x.x || x.y || x.z || x.w;
//}
//
//// all
//GLM_FUNC_QUALIFIER bool all(const tvec2<bool>& x)
//{
//    return x.x && x.y;
//}
//
//GLM_FUNC_QUALIFIER bool all(const tvec3<bool>& x)
//{
//    return x.x && x.y && x.z;
//}
//
//GLM_FUNC_QUALIFIER bool all(const tvec4<bool>& x)
//{
//    return x.x && x.y && x.z && x.w;
//}
//
//// not
//GLM_FUNC_QUALIFIER tvec2<bool>::bool_type not_
//(
//	tvec2<bool> const & v
//)
//{
//    return tvec2<bool>::bool_type(!v.x, !v.y);
//}
//
//GLM_FUNC_QUALIFIER tvec3<bool>::bool_type not_
//(
//	tvec3<bool> const & v
//)
//{
//    return tvec3<bool>::bool_type(!v.x, !v.y, !v.z);
//}
//
//GLM_FUNC_QUALIFIER tvec4<bool>::bool_type not_
//(
//	tvec4<bool> const & v
//)
//{
//    return tvec4<bool>::bool_type(!v.x, !v.y, !v.z, !v.w);
//}