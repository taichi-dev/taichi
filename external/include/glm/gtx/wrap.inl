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
/// @ref gtx_wrap
/// @file glm/gtx/wrap.inl
/// @date 2009-11-25 / 2011-06-07
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

namespace glm
{
	template <typename genType> 
	GLM_FUNC_QUALIFIER genType clamp
	(
		genType const & Texcoord
	)
	{
		return glm::clamp(Texcoord, genType(0), genType(1));
	}

	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tvec2<T, P> clamp
	(
		tvec2<T, P> const & Texcoord
	)
	{
		tvec2<T, P> Result;
		for(typename tvec2<T, P>::size_type i = 0; i < tvec2<T, P>::value_size(); ++i)
			Result[i] = clamp(Texcoord[i]);
		return Result;
	}

	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tvec3<T, P> clamp
	(
		tvec3<T, P> const & Texcoord
	)
	{
		tvec3<T, P> Result;
		for(typename tvec3<T, P>::size_type i = 0; i < tvec3<T, P>::value_size(); ++i)
			Result[i] = clamp(Texcoord[i]);
		return Result;
	}

	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tvec4<T, P> clamp
	(
		tvec4<T, P> const & Texcoord
	)
	{
		tvec4<T, P> Result;
		for(typename tvec4<T, P>::size_type i = 0; i < tvec4<T, P>::value_size(); ++i)
			Result[i] = clamp(Texcoord[i]);
		return Result;
	}

	////////////////////////
	// repeat

	template <typename genType> 
	GLM_FUNC_QUALIFIER genType repeat
	(
		genType const & Texcoord
	)
	{
		return glm::fract(Texcoord);
	}

	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tvec2<T, P> repeat
	(
		tvec2<T, P> const & Texcoord
	)
	{
		tvec2<T, P> Result;
		for(typename tvec2<T, P>::size_type i = 0; i < tvec2<T, P>::value_size(); ++i)
			Result[i] = repeat(Texcoord[i]);
		return Result;
	}

	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tvec3<T, P> repeat
	(
		tvec3<T, P> const & Texcoord
	)
	{
		tvec3<T, P> Result;
		for(typename tvec3<T, P>::size_type i = 0; i < tvec3<T, P>::value_size(); ++i)
			Result[i] = repeat(Texcoord[i]);
		return Result;
	}

	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tvec4<T, P> repeat
	(
		tvec4<T, P> const & Texcoord
	)
	{
		tvec4<T, P> Result;
		for(typename tvec4<T, P>::size_type i = 0; i < tvec4<T, P>::value_size(); ++i)
			Result[i] = repeat(Texcoord[i]);
		return Result;
	}

	////////////////////////
	// mirrorRepeat

	template <typename genType, precision P> 
	GLM_FUNC_QUALIFIER genType mirrorRepeat
	(
		genType const & Texcoord
	)
	{
		genType const Clamp = genType(int(glm::floor(Texcoord)) % 2);
		genType const Floor = glm::floor(Texcoord);
		genType const Rest = Texcoord - Floor;
		genType const Mirror = Clamp + Rest;

		genType Out;
		if(Mirror >= genType(1))
			Out = genType(1) - Rest;
		else
			Out = Rest;
		return Out;
	}

	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tvec2<T, P> mirrorRepeat
	(
		tvec2<T, P> const & Texcoord
	)
	{
		tvec2<T, P> Result;
		for(typename tvec2<T, P>::size_type i = 0; i < tvec2<T, P>::value_size(); ++i)
			Result[i] = mirrorRepeat(Texcoord[i]);
		return Result;
	}

	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tvec3<T, P> mirrorRepeat
	(
		tvec3<T, P> const & Texcoord
	)
	{
		tvec3<T, P> Result;
		for(typename tvec3<T, P>::size_type i = 0; i < tvec3<T, P>::value_size(); ++i)
			Result[i] = mirrorRepeat(Texcoord[i]);
		return Result;
	}

	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tvec4<T, P> mirrorRepeat
	(
		tvec4<T, P> const & Texcoord
	)
	{
		tvec4<T, P> Result;
		for(typename tvec4<T, P>::size_type i = 0; i < tvec4<T, P>::value_size(); ++i)
			Result[i] = mirrorRepeat(Texcoord[i]);
		return Result;
	}
}//namespace glm
