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
/// @ref gtx_matrix_decompose
/// @file glm/gtx/matrix_decompose.inl
/// @date 2014-08-29 / 2014-08-29
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

namespace glm
{
	/// Make a linear combination of two vectors and return the result.
	// result = (a * ascl) + (b * bscl)
	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec3<T, P> combine(
		tvec3<T, P> const & a, 
		tvec3<T, P> const & b,
		T ascl, T bscl)
	{
		return (a * ascl) + (b * bscl);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER void v3Scale(tvec3<T, P> & v, T desiredLength)
	{
		T len = glm::length(v);
		if(len != 0)
		{
			T l = desiredLength / len;
			v[0] *= l;
			v[1] *= l;
			v[2] *= l;
		}
	}

	/**
	* Matrix decompose
	* http://www.opensource.apple.com/source/WebCore/WebCore-514/platform/graphics/transforms/TransformationMatrix.cpp
	* Decomposes the mode matrix to translations,rotation scale components
	* 
	*/

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER bool decompose(tmat4x4<T, P> const & ModelMatrix, tvec3<T, P> & Scale, tquat<T, P> & Orientation, tvec3<T, P> & Translation, tvec3<T, P> & Skew, tvec4<T, P> & Perspective)
	{
		tmat4x4<T, P> LocalMatrix(ModelMatrix);

		// Normalize the matrix.
		if(LocalMatrix[3][3] == static_cast<T>(0))
			return false;

		for(length_t i = 0; i < 4; ++i)
		for(length_t j = 0; j < 4; ++j)
			LocalMatrix[i][j] /= LocalMatrix[3][3];

		// perspectiveMatrix is used to solve for perspective, but it also provides
		// an easy way to test for singularity of the upper 3x3 component.
		tmat4x4<T, P> PerspectiveMatrix(LocalMatrix);

		for(length_t i = 0; i < 3; i++)
			PerspectiveMatrix[i][3] = 0;
		PerspectiveMatrix[3][3] = 1;

		/// TODO: Fixme!
		if(determinant(PerspectiveMatrix) == static_cast<T>(0))
			return false;

		// First, isolate perspective.  This is the messiest.
		if(LocalMatrix[0][3] != 0 || LocalMatrix[1][3] != 0 || LocalMatrix[2][3] != 0)
		{
			// rightHandSide is the right hand side of the equation.
			tvec4<T, P> RightHandSide;
			RightHandSide[0] = LocalMatrix[0][3];
			RightHandSide[1] = LocalMatrix[1][3];
			RightHandSide[2] = LocalMatrix[2][3];
			RightHandSide[3] = LocalMatrix[3][3];

			// Solve the equation by inverting PerspectiveMatrix and multiplying
			// rightHandSide by the inverse.  (This is the easiest way, not
			// necessarily the best.)
			tmat4x4<T, P> InversePerspectiveMatrix = glm::inverse(PerspectiveMatrix);//   inverse(PerspectiveMatrix, inversePerspectiveMatrix);
			tmat4x4<T, P> TransposedInversePerspectiveMatrix = glm::transpose(InversePerspectiveMatrix);//   transposeMatrix4(inversePerspectiveMatrix, transposedInversePerspectiveMatrix);

			Perspective = TransposedInversePerspectiveMatrix * RightHandSide;
			//  v4MulPointByMatrix(rightHandSide, transposedInversePerspectiveMatrix, perspectivePoint);

			// Clear the perspective partition
			LocalMatrix[0][3] = LocalMatrix[1][3] = LocalMatrix[2][3] = 0;
			LocalMatrix[3][3] = 1;
		}
		else
		{
			// No perspective.
			Perspective = tvec4<T, P>(0, 0, 0, 1);
		}

		// Next take care of translation (easy).
		Translation = tvec3<T, P>(LocalMatrix[3]);
		LocalMatrix[3] = tvec4<T, P>(0, 0, 0, LocalMatrix[3].w);

		tvec3<T, P> Row[3], Pdum3;

		// Now get scale and shear.
		for(length_t i = 0; i < 3; ++i)
			for(int j = 0; j < 3; ++j)
				Row[i][j] = LocalMatrix[i][j];

		// Compute X scale factor and normalize first row.
		Scale.x = length(Row[0]);// v3Length(Row[0]);

		v3Scale(Row[0], static_cast<T>(1));

		// Compute XY shear factor and make 2nd row orthogonal to 1st.
		Skew.z = dot(Row[0], Row[1]);
		Row[1] = combine(Row[1], Row[0], static_cast<T>(1), -Skew.z);

		// Now, compute Y scale and normalize 2nd row.
		Scale.y = length(Row[1]);
		v3Scale(Row[1], static_cast<T>(1));
		Skew.z /= Scale.y;

		// Compute XZ and YZ shears, orthogonalize 3rd row.
		Skew.y = glm::dot(Row[0], Row[2]);
		Row[2] = combine(Row[2], Row[0], static_cast<T>(1), -Skew.y);
		Skew.x = glm::dot(Row[1], Row[2]);
		Row[2] = combine(Row[2], Row[1], static_cast<T>(1), -Skew.x);

		// Next, get Z scale and normalize 3rd row.
		Scale.z = length(Row[2]);
		v3Scale(Row[2], static_cast<T>(1));
		Skew.y /= Scale.z;
		Skew.x /= Scale.z;

		// At this point, the matrix (in rows[]) is orthonormal.
		// Check for a coordinate system flip.  If the determinant
		// is -1, then negate the matrix and the scaling factors.
		Pdum3 = cross(Row[1], Row[2]); // v3Cross(row[1], row[2], Pdum3);
		if(dot(Row[0], Pdum3) < 0)
		{
			for(length_t i = 0; i < 3; i++)
			{
				Scale.x *= static_cast<T>(-1);
				Row[i] *= static_cast<T>(-1);
			}
		}

		// Now, get the rotations out, as described in the gem.

		// FIXME - Add the ability to return either quaternions (which are
		// easier to recompose with) or Euler angles (rx, ry, rz), which
		// are easier for authors to deal with. The latter will only be useful
		// when we fix https://bugs.webkit.org/show_bug.cgi?id=23799, so I
		// will leave the Euler angle code here for now.

		// ret.rotateY = asin(-Row[0][2]);
		// if (cos(ret.rotateY) != 0) {
		//     ret.rotateX = atan2(Row[1][2], Row[2][2]);
		//     ret.rotateZ = atan2(Row[0][1], Row[0][0]);
		// } else {
		//     ret.rotateX = atan2(-Row[2][0], Row[1][1]);
		//     ret.rotateZ = 0;
		// }

		T s, t, x, y, z, w;

		t = Row[0][0] + Row[1][1] + Row[2][2] + 1.0;

		if(t > 1e-4)
		{
			s = 0.5 / sqrt(t);
			w = 0.25 / s;
			x = (Row[2][1] - Row[1][2]) * s;
			y = (Row[0][2] - Row[2][0]) * s;
			z = (Row[1][0] - Row[0][1]) * s;
		}
		else if(Row[0][0] > Row[1][1] && Row[0][0] > Row[2][2])
		{ 
			s = sqrt (1.0 + Row[0][0] - Row[1][1] - Row[2][2]) * 2.0; // S=4*qx 
			x = 0.25 * s;
			y = (Row[0][1] + Row[1][0]) / s; 
			z = (Row[0][2] + Row[2][0]) / s; 
			w = (Row[2][1] - Row[1][2]) / s;
		}
		else if(Row[1][1] > Row[2][2])
		{ 
			s = sqrt (1.0 + Row[1][1] - Row[0][0] - Row[2][2]) * 2.0; // S=4*qy
			x = (Row[0][1] + Row[1][0]) / s; 
			y = 0.25 * s;
			z = (Row[1][2] + Row[2][1]) / s; 
			w = (Row[0][2] - Row[2][0]) / s;
		}
		else
		{ 
			s = sqrt(1.0 + Row[2][2] - Row[0][0] - Row[1][1]) * 2.0; // S=4*qz
			x = (Row[0][2] + Row[2][0]) / s;
			y = (Row[1][2] + Row[2][1]) / s; 
			z = 0.25 * s;
			w = (Row[1][0] - Row[0][1]) / s;
		}

		Orientation.x = x;
		Orientation.y = y;
		Orientation.z = z;
		Orientation.w = w;

		return true;
	}
}//namespace glm
