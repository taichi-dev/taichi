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
/// @ref gtx_string_cast
/// @file glm/gtx/string_cast.inl
/// @date 2008-04-26 / 2014-05-10
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

#include <cstdarg>
#include <cstdio>

namespace glm{
namespace detail
{
	GLM_FUNC_QUALIFIER std::string format(const char* msg, ...)
	{
		std::size_t const STRING_BUFFER(4096);
		char text[STRING_BUFFER];
		va_list list;

		if(msg == 0)
			return std::string();

		va_start(list, msg);
#		if(GLM_COMPILER & GLM_COMPILER_VC)
			vsprintf_s(text, STRING_BUFFER, msg, list);
#		else//
			vsprintf(text, msg, list);
#		endif//
		va_end(list);

		return std::string(text);
	}

	static const char* LabelTrue = "true";
	static const char* LabelFalse = "false";

	template <typename T, bool isFloat = false>
	struct literal
	{
		GLM_FUNC_QUALIFIER static char const * value() {return "%d";};
	};

	template <typename T>
	struct literal<T, true>
	{
		GLM_FUNC_QUALIFIER static char const * value() {return "%f";};
	};

#	if GLM_MODEL == GLM_MODEL_32 && GLM_COMPILER && GLM_COMPILER_VC
	template <>
	struct literal<uint64_t, false>
	{
		GLM_FUNC_QUALIFIER static char const * value() {return "%lld";};
	};

	template <>
	struct literal<int64_t, false>
	{
		GLM_FUNC_QUALIFIER static char const * value() {return "%lld";};
	};
#	endif//GLM_MODEL == GLM_MODEL_32 && GLM_COMPILER && GLM_COMPILER_VC

	template <typename T>
	struct prefix{};

	template <>
	struct prefix<float>
	{
		GLM_FUNC_QUALIFIER static char const * value() {return "";};
	};

	template <>
	struct prefix<double>
	{
		GLM_FUNC_QUALIFIER static char const * value() {return "d";};
	};

	template <>
	struct prefix<bool>
	{
		GLM_FUNC_QUALIFIER static char const * value() {return "b";};
	};

	template <>
	struct prefix<uint8_t>
	{
		GLM_FUNC_QUALIFIER static char const * value() {return "u8";};
	};

	template <>
	struct prefix<int8_t>
	{
		GLM_FUNC_QUALIFIER static char const * value() {return "i8";};
	};

	template <>
	struct prefix<uint16_t>
	{
		GLM_FUNC_QUALIFIER static char const * value() {return "u16";};
	};

	template <>
	struct prefix<int16_t>
	{
		GLM_FUNC_QUALIFIER static char const * value() {return "i16";};
	};

	template <>
	struct prefix<uint32_t>
	{
		GLM_FUNC_QUALIFIER static char const * value() {return "u";};
	};

	template <>
	struct prefix<int32_t>
	{
		GLM_FUNC_QUALIFIER static char const * value() {return "i";};
	};

	template <>
	struct prefix<uint64_t>
	{
		GLM_FUNC_QUALIFIER static char const * value() {return "u64";};
	};

	template <>
	struct prefix<int64_t>
	{
		GLM_FUNC_QUALIFIER static char const * value() {return "i64";};
	};

	template <template <typename, precision> class matType, typename T, precision P>
	struct compute_to_string
	{};

	template <precision P>
	struct compute_to_string<tvec1, bool, P>
	{
		GLM_FUNC_QUALIFIER static std::string call(tvec1<bool, P> const & x)
		{
			return detail::format("bvec1(%s)",
				x[0] ? detail::LabelTrue : detail::LabelFalse);
		}
	};

	template <precision P>
	struct compute_to_string<tvec2, bool, P>
	{
		GLM_FUNC_QUALIFIER static std::string call(tvec2<bool, P> const & x)
		{
			return detail::format("bvec2(%s, %s)",
				x[0] ? detail::LabelTrue : detail::LabelFalse,
				x[1] ? detail::LabelTrue : detail::LabelFalse);
		}
	};

	template <precision P>
	struct compute_to_string<tvec3, bool, P>
	{
		GLM_FUNC_QUALIFIER static std::string call(tvec3<bool, P> const & x)
		{
			return detail::format("bvec3(%s, %s, %s)",
				x[0] ? detail::LabelTrue : detail::LabelFalse,
				x[1] ? detail::LabelTrue : detail::LabelFalse,
				x[2] ? detail::LabelTrue : detail::LabelFalse);
		}
	};

	template <precision P>
	struct compute_to_string<tvec4, bool, P>
	{
		GLM_FUNC_QUALIFIER static std::string call(tvec4<bool, P> const & x)
		{
			return detail::format("bvec4(%s, %s, %s, %s)",
				x[0] ? detail::LabelTrue : detail::LabelFalse,
				x[1] ? detail::LabelTrue : detail::LabelFalse,
				x[2] ? detail::LabelTrue : detail::LabelFalse,
				x[3] ? detail::LabelTrue : detail::LabelFalse);
		}
	};

	template <typename T, precision P>
	struct compute_to_string<tvec1, T, P>
	{
		GLM_FUNC_QUALIFIER static std::string call(tvec1<T, P> const & x)
		{
			char const * PrefixStr = prefix<T>::value();
			char const * LiteralStr = literal<T, std::numeric_limits<T>::is_iec559>::value();
			std::string FormatStr(detail::format("%svec1(%s)",
				PrefixStr,
				LiteralStr));

			return detail::format(FormatStr.c_str(), x[0]);
		}
	};

	template <typename T, precision P>
	struct compute_to_string<tvec2, T, P>
	{
		GLM_FUNC_QUALIFIER static std::string call(tvec2<T, P> const & x)
		{
			char const * PrefixStr = prefix<T>::value();
			char const * LiteralStr = literal<T, std::numeric_limits<T>::is_iec559>::value();
			std::string FormatStr(detail::format("%svec2(%s, %s)",
				PrefixStr,
				LiteralStr, LiteralStr));

			return detail::format(FormatStr.c_str(), x[0], x[1]);
		}
	};

	template <typename T, precision P>
	struct compute_to_string<tvec3, T, P>
	{
		GLM_FUNC_QUALIFIER static std::string call(tvec3<T, P> const & x)
		{
			char const * PrefixStr = prefix<T>::value();
			char const * LiteralStr = literal<T, std::numeric_limits<T>::is_iec559>::value();
			std::string FormatStr(detail::format("%svec3(%s, %s, %s)",
				PrefixStr,
				LiteralStr, LiteralStr, LiteralStr));

			return detail::format(FormatStr.c_str(), x[0], x[1], x[2]);
		}
	};

	template <typename T, precision P>
	struct compute_to_string<tvec4, T, P>
	{
		GLM_FUNC_QUALIFIER static std::string call(tvec4<T, P> const & x)
		{
			char const * PrefixStr = prefix<T>::value();
			char const * LiteralStr = literal<T, std::numeric_limits<T>::is_iec559>::value();
			std::string FormatStr(detail::format("%svec4(%s, %s, %s, %s)",
				PrefixStr,
				LiteralStr, LiteralStr, LiteralStr, LiteralStr));

			return detail::format(FormatStr.c_str(), x[0], x[1], x[2], x[3]);
		}
	};


	template <typename T, precision P>
	struct compute_to_string<tmat2x2, T, P>
	{
		GLM_FUNC_QUALIFIER static std::string call(tmat2x2<T, P> const & x)
		{
			char const * PrefixStr = prefix<T>::value();
			char const * LiteralStr = literal<T, std::numeric_limits<T>::is_iec559>::value();
			std::string FormatStr(detail::format("%smat2x2((%s, %s), (%s, %s))",
				PrefixStr,
				LiteralStr, LiteralStr,
				LiteralStr, LiteralStr));

			return detail::format(FormatStr.c_str(),
				x[0][0], x[0][1],
				x[1][0], x[1][1]);
		}
	};

	template <typename T, precision P>
	struct compute_to_string<tmat2x3, T, P>
	{
		GLM_FUNC_QUALIFIER static std::string call(tmat2x3<T, P> const & x)
		{
			char const * PrefixStr = prefix<T>::value();
			char const * LiteralStr = literal<T, std::numeric_limits<T>::is_iec559>::value();
			std::string FormatStr(detail::format("%smat2x3((%s, %s, %s), (%s, %s, %s))",
				PrefixStr,
				LiteralStr, LiteralStr, LiteralStr,
				LiteralStr, LiteralStr, LiteralStr));

			return detail::format(FormatStr.c_str(),
				x[0][0], x[0][1], x[0][2],
				x[1][0], x[1][1], x[1][2]);
		}
	};

	template <typename T, precision P>
	struct compute_to_string<tmat2x4, T, P>
	{
		GLM_FUNC_QUALIFIER static std::string call(tmat2x4<T, P> const & x)
		{
			char const * PrefixStr = prefix<T>::value();
			char const * LiteralStr = literal<T, std::numeric_limits<T>::is_iec559>::value();
			std::string FormatStr(detail::format("%smat2x4((%s, %s, %s, %s), (%s, %s, %s, %s))",
				PrefixStr,
				LiteralStr, LiteralStr, LiteralStr, LiteralStr,
				LiteralStr, LiteralStr, LiteralStr, LiteralStr));

			return detail::format(FormatStr.c_str(),
				x[0][0], x[0][1], x[0][2], x[0][3],
				x[1][0], x[1][1], x[1][2], x[1][3]);
		}
	};

	template <typename T, precision P>
	struct compute_to_string<tmat3x2, T, P>
	{
		GLM_FUNC_QUALIFIER static std::string call(tmat3x2<T, P> const & x)
		{
			char const * PrefixStr = prefix<T>::value();
			char const * LiteralStr = literal<T, std::numeric_limits<T>::is_iec559>::value();
			std::string FormatStr(detail::format("%smat3x2((%s, %s), (%s, %s), (%s, %s))",
				PrefixStr,
				LiteralStr, LiteralStr,
				LiteralStr, LiteralStr,
				LiteralStr, LiteralStr));

			return detail::format(FormatStr.c_str(),
				x[0][0], x[0][1],
				x[1][0], x[1][1],
				x[2][0], x[2][1]);
		}
	};

	template <typename T, precision P>
	struct compute_to_string<tmat3x3, T, P>
	{
		GLM_FUNC_QUALIFIER static std::string call(tmat3x3<T, P> const & x)
		{
			char const * PrefixStr = prefix<T>::value();
			char const * LiteralStr = literal<T, std::numeric_limits<T>::is_iec559>::value();
			std::string FormatStr(detail::format("%smat3x3((%s, %s, %s), (%s, %s, %s), (%s, %s, %s))",
				PrefixStr,
				LiteralStr, LiteralStr, LiteralStr,
				LiteralStr, LiteralStr, LiteralStr,
				LiteralStr, LiteralStr, LiteralStr));

			return detail::format(FormatStr.c_str(),
				x[0][0], x[0][1], x[0][2],
				x[1][0], x[1][1], x[1][2],
				x[2][0], x[2][1], x[2][2]);
		}
	};

	template <typename T, precision P>
	struct compute_to_string<tmat3x4, T, P>
	{
		GLM_FUNC_QUALIFIER static std::string call(tmat3x4<T, P> const & x)
		{
			char const * PrefixStr = prefix<T>::value();
			char const * LiteralStr = literal<T, std::numeric_limits<T>::is_iec559>::value();
			std::string FormatStr(detail::format("%smat3x4((%s, %s, %s, %s), (%s, %s, %s, %s), (%s, %s, %s, %s))",
				PrefixStr,
				LiteralStr, LiteralStr, LiteralStr, LiteralStr,
				LiteralStr, LiteralStr, LiteralStr, LiteralStr,
				LiteralStr, LiteralStr, LiteralStr, LiteralStr));

			return detail::format(FormatStr.c_str(),
				x[0][0], x[0][1], x[0][2], x[0][3],
				x[1][0], x[1][1], x[1][2], x[1][3],
				x[2][0], x[2][1], x[2][2], x[2][3]);
		}
	};

	template <typename T, precision P>
	struct compute_to_string<tmat4x2, T, P>
	{
		GLM_FUNC_QUALIFIER static std::string call(tmat4x2<T, P> const & x)
		{
			char const * PrefixStr = prefix<T>::value();
			char const * LiteralStr = literal<T, std::numeric_limits<T>::is_iec559>::value();
			std::string FormatStr(detail::format("%smat4x2((%s, %s), (%s, %s), (%s, %s), (%s, %s))",
				PrefixStr,
				LiteralStr, LiteralStr,
				LiteralStr, LiteralStr,
				LiteralStr, LiteralStr,
				LiteralStr, LiteralStr));

			return detail::format(FormatStr.c_str(),
				x[0][0], x[0][1],
				x[1][0], x[1][1],
				x[2][0], x[2][1],
				x[3][0], x[3][1]);
		}
	};

	template <typename T, precision P>
	struct compute_to_string<tmat4x3, T, P>
	{
		GLM_FUNC_QUALIFIER static std::string call(tmat4x3<T, P> const & x)
		{
			char const * PrefixStr = prefix<T>::value();
			char const * LiteralStr = literal<T, std::numeric_limits<T>::is_iec559>::value();
			std::string FormatStr(detail::format("%smat4x3((%s, %s, %s), (%s, %s, %s), (%s, %s, %s), (%s, %s, %s))",
				PrefixStr,
				LiteralStr, LiteralStr, LiteralStr,
				LiteralStr, LiteralStr, LiteralStr,
				LiteralStr, LiteralStr, LiteralStr,
				LiteralStr, LiteralStr, LiteralStr));

			return detail::format(FormatStr.c_str(),
				x[0][0], x[0][1], x[0][2],
				x[1][0], x[1][1], x[1][2],
				x[2][0], x[2][1], x[2][2],
				x[3][0], x[3][1], x[3][2]);
		}
	};

	template <typename T, precision P>
	struct compute_to_string<tmat4x4, T, P>
	{
		GLM_FUNC_QUALIFIER static std::string call(tmat4x4<T, P> const & x)
		{
			char const * PrefixStr = prefix<T>::value();
			char const * LiteralStr = literal<T, std::numeric_limits<T>::is_iec559>::value();
			std::string FormatStr(detail::format("%smat4x4((%s, %s, %s, %s), (%s, %s, %s, %s), (%s, %s, %s, %s), (%s, %s, %s, %s))",
				PrefixStr,
				LiteralStr, LiteralStr, LiteralStr, LiteralStr,
				LiteralStr, LiteralStr, LiteralStr, LiteralStr,
				LiteralStr, LiteralStr, LiteralStr, LiteralStr,
				LiteralStr, LiteralStr, LiteralStr, LiteralStr));

			return detail::format(FormatStr.c_str(),
				x[0][0], x[0][1], x[0][2], x[0][3],
				x[1][0], x[1][1], x[1][2], x[1][3],
				x[2][0], x[2][1], x[2][2], x[2][3],
				x[3][0], x[3][1], x[3][2], x[3][3]);
		}
	};
}//namespace detail

template <template <typename, precision> class matType, typename T, precision P>
GLM_FUNC_DECL std::string to_string(matType<T, P> const & x)
{
	return detail::compute_to_string<matType, T, P>::call(x);
}

}//namespace glm
