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
/// @ref gtx_matrix_interpolation
/// @file glm/gtx/matrix_interpolation.hpp
/// @date 2011-03-05 / 2011-03-05
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

namespace glm
{
	template <typename T, precision P>
	GLM_FUNC_QUALIFIER void axisAngle
	(
		tmat4x4<T, P> const & mat,
		tvec3<T, P> & axis,
		T & angle
	)
	{
		T epsilon = (T)0.01;
		T epsilon2 = (T)0.1;

		if((abs(mat[1][0] - mat[0][1]) < epsilon) && (abs(mat[2][0] - mat[0][2]) < epsilon) && (abs(mat[2][1] - mat[1][2]) < epsilon))
		{
			if ((abs(mat[1][0] + mat[0][1]) < epsilon2) && (abs(mat[2][0] + mat[0][2]) < epsilon2) && (abs(mat[2][1] + mat[1][2]) < epsilon2) && (abs(mat[0][0] + mat[1][1] + mat[2][2] - (T)3.0) < epsilon2))
			{
				angle = (T)0.0;
				axis.x = (T)1.0;
				axis.y = (T)0.0;
				axis.z = (T)0.0;
				return;
			}
			angle = static_cast<T>(3.1415926535897932384626433832795);
			T xx = (mat[0][0] + (T)1.0) / (T)2.0;
			T yy = (mat[1][1] + (T)1.0) / (T)2.0;
			T zz = (mat[2][2] + (T)1.0) / (T)2.0;
			T xy = (mat[1][0] + mat[0][1]) / (T)4.0;
			T xz = (mat[2][0] + mat[0][2]) / (T)4.0;
			T yz = (mat[2][1] + mat[1][2]) / (T)4.0;
			if((xx > yy) && (xx > zz))
			{
				if (xx < epsilon) {
					axis.x = (T)0.0;
					axis.y = (T)0.7071;
					axis.z = (T)0.7071;
				} else {
					axis.x = sqrt(xx);
					axis.y = xy / axis.x;
					axis.z = xz / axis.x;
				}
			}
			else if (yy > zz)
			{
				if (yy < epsilon) {
					axis.x = (T)0.7071;
					axis.y = (T)0.0;
					axis.z = (T)0.7071;
				} else {
					axis.y = sqrt(yy);
					axis.x = xy / axis.y;
					axis.z = yz / axis.y;
				}
			}
			else
			{
				if (zz < epsilon) {
					axis.x = (T)0.7071;
					axis.y = (T)0.7071;
					axis.z = (T)0.0;
				} else {
					axis.z = sqrt(zz);
					axis.x = xz / axis.z;
					axis.y = yz / axis.z;
				}
			}
			return;
		}
		T s = sqrt((mat[2][1] - mat[1][2]) * (mat[2][1] - mat[1][2]) + (mat[2][0] - mat[0][2]) * (mat[2][0] - mat[0][2]) + (mat[1][0] - mat[0][1]) * (mat[1][0] - mat[0][1]));
		if (glm::abs(s) < T(0.001))
			s = (T)1.0;
		angle = acos((mat[0][0] + mat[1][1] + mat[2][2] - (T)1.0) / (T)2.0);
		axis.x = (mat[1][2] - mat[2][1]) / s;
		axis.y = (mat[2][0] - mat[0][2]) / s;
		axis.z = (mat[0][1] - mat[1][0]) / s;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat4x4<T, P> axisAngleMatrix
	(
		tvec3<T, P> const & axis,
		T const angle
	)
	{
		T c = cos(angle);
		T s = sin(angle);
		T t = static_cast<T>(1) - c;
		tvec3<T, P> n = normalize(axis);

		return tmat4x4<T, P>(
			t * n.x * n.x + c,          t * n.x * n.y + n.z * s,    t * n.x * n.z - n.y * s,    T(0),
			t * n.x * n.y - n.z * s,    t * n.y * n.y + c,          t * n.y * n.z + n.x * s,    T(0),
			t * n.x * n.z + n.y * s,    t * n.y * n.z - n.x * s,    t * n.z * n.z + c,          T(0),
			T(0),                        T(0),                        T(0),                     T(1)
		);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat4x4<T, P> extractMatrixRotation
	(
		tmat4x4<T, P> const & mat
	)
	{
		return tmat4x4<T, P>(
			mat[0][0], mat[0][1], mat[0][2], 0.0,
			mat[1][0], mat[1][1], mat[1][2], 0.0,
			mat[2][0], mat[2][1], mat[2][2], 0.0,
			0.0,       0.0,       0.0,       1.0
		);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat4x4<T, P> interpolate
	(
		tmat4x4<T, P> const & m1,
		tmat4x4<T, P> const & m2,
		T const delta
	)
	{
		tmat4x4<T, P> m1rot = extractMatrixRotation(m1);
		tmat4x4<T, P> dltRotation = m2 * transpose(m1rot);
		tvec3<T, P> dltAxis;
		T dltAngle;
		axisAngle(dltRotation, dltAxis, dltAngle);
		tmat4x4<T, P> out = axisAngleMatrix(dltAxis, dltAngle * delta) * m1rot;
		out[3][0] = m1[3][0] + delta * (m2[3][0] - m1[3][0]);
		out[3][1] = m1[3][1] + delta * (m2[3][1] - m1[3][1]);
		out[3][2] = m1[3][2] + delta * (m2[3][2] - m1[3][2]);
		return out;
	}
}//namespace glm
