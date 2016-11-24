///////////////////////////////////////////////////////////////////////////////////
/// OpenGL Mathematics (glm.g-truc.net)
///
/// Copyright (c) 2005 - 2013 G-Truc Creation (www.g-truc.net)
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
/// @ref gtx_io
/// @file glm/gtx/io.hpp
/// @date 2013-11-22 / 2014-11-25
/// @author Jan P Springer (regnirpsj@gmail.com)
///
/// @see core (dependence)
/// @see gtc_quaternion (dependence)
///
/// @defgroup gtx_io GLM_GTX_io
/// @ingroup gtx
/// 
/// @brief std::[w]ostream support for glm types
///
///        std::[w]ostream support for glm types + precision/width/etc. manipulators
///        based on howard hinnant's std::chrono io proposal
///        [http://home.roadrunner.com/~hinnant/bloomington/chrono_io.html]
///
/// <glm/gtx/io.hpp> needs to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependency:
#include "../glm.hpp"
#include "../gtx/quaternion.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
# pragma message("GLM: GLM_GTX_io extension included")
#endif

#include <iosfwd>  // std::basic_ostream<> (fwd)
#include <locale>  // std::locale, std::locale::facet, std::locale::id
#include <utility> // std::pair<>

namespace glm
{
	/// @addtogroup gtx_io
	/// @{

	namespace io
	{
		enum order_type { column_major, row_major};

		template <typename CTy>
		class format_punct : public std::locale::facet
		{
			typedef CTy char_type;

		public:

			static std::locale::id id;

			bool       formatted;
			unsigned   precision;
			unsigned   width;
			char_type  separator;
			char_type  delim_left;
			char_type  delim_right;
			char_type  space;
			char_type  newline;
			order_type order;

			explicit format_punct(size_t a = 0);
			explicit format_punct(format_punct const&);
		};

		template <typename CTy, typename CTr = std::char_traits<CTy> >
		class basic_state_saver {

		public:

			explicit basic_state_saver(std::basic_ios<CTy,CTr>&);
					~basic_state_saver();

		private:

			typedef ::std::basic_ios<CTy,CTr>      state_type;
			typedef typename state_type::char_type char_type;
			typedef ::std::ios_base::fmtflags      flags_type;
			typedef ::std::streamsize              streamsize_type;
			typedef ::std::locale const            locale_type;

			state_type&     state_;
			flags_type      flags_;
			streamsize_type precision_;
			streamsize_type width_;
			char_type       fill_;
			locale_type     locale_;

			basic_state_saver& operator=(basic_state_saver const&);
		};

		typedef basic_state_saver<char>     state_saver;
		typedef basic_state_saver<wchar_t> wstate_saver;

		template <typename CTy, typename CTr = std::char_traits<CTy> >
		class basic_format_saver
		{
		public:

			explicit basic_format_saver(std::basic_ios<CTy,CTr>&);
					~basic_format_saver();

		private:

			basic_state_saver<CTy> const bss_;

			basic_format_saver& operator=(basic_format_saver const&);
		};

		typedef basic_format_saver<char>     format_saver;
		typedef basic_format_saver<wchar_t> wformat_saver;

		struct precision
		{
			unsigned value;

			explicit precision(unsigned);
		};

		struct width
		{
			unsigned value;

			explicit width(unsigned);
		};

		template <typename CTy>
		struct delimeter
		{
			CTy value[3];

			explicit delimeter(CTy /* left */, CTy /* right */, CTy /* separator */ = ',');
		};

		struct order
		{
			order_type value;

			explicit order(order_type);
		};

		// functions, inlined (inline)

		template <typename FTy, typename CTy, typename CTr>
		FTy const& get_facet(std::basic_ios<CTy,CTr>&);
		template <typename FTy, typename CTy, typename CTr>
		std::basic_ios<CTy,CTr>& formatted(std::basic_ios<CTy,CTr>&);
		template <typename FTy, typename CTy, typename CTr>
		std::basic_ios<CTy,CTr>& unformattet(std::basic_ios<CTy,CTr>&);

		template <typename CTy, typename CTr>
		std::basic_ostream<CTy, CTr>& operator<<(std::basic_ostream<CTy, CTr>&, precision const&);
		template <typename CTy, typename CTr>
		std::basic_ostream<CTy, CTr>& operator<<(std::basic_ostream<CTy, CTr>&, width const&);
		template <typename CTy, typename CTr>
		std::basic_ostream<CTy, CTr>& operator<<(std::basic_ostream<CTy, CTr>&, delimeter<CTy> const&);
		template <typename CTy, typename CTr>
		std::basic_ostream<CTy, CTr>& operator<<(std::basic_ostream<CTy, CTr>&, order const&);
	}//namespace io

	template <typename CTy, typename CTr, typename T, precision P>
	GLM_FUNC_DECL std::basic_ostream<CTy,CTr>& operator<<(std::basic_ostream<CTy,CTr>&, tquat<T,P> const&);
	template <typename CTy, typename CTr, typename T, precision P>
	GLM_FUNC_DECL std::basic_ostream<CTy,CTr>& operator<<(std::basic_ostream<CTy,CTr>&, tvec2<T,P> const&);
	template <typename CTy, typename CTr, typename T, precision P>
	GLM_FUNC_DECL std::basic_ostream<CTy,CTr>& operator<<(std::basic_ostream<CTy,CTr>&, tvec3<T,P> const&);
	template <typename CTy, typename CTr, typename T, precision P>
	GLM_FUNC_DECL std::basic_ostream<CTy,CTr>& operator<<(std::basic_ostream<CTy,CTr>&, tvec4<T,P> const&);
	template <typename CTy, typename CTr, typename T, precision P>
	GLM_FUNC_DECL std::basic_ostream<CTy,CTr>& operator<<(std::basic_ostream<CTy,CTr>&, tmat2x2<T,P> const&);
	template <typename CTy, typename CTr, typename T, precision P>
	GLM_FUNC_DECL std::basic_ostream<CTy,CTr>& operator<<(std::basic_ostream<CTy,CTr>&, tmat2x3<T,P> const&);
	template <typename CTy, typename CTr, typename T, precision P>
	GLM_FUNC_DECL std::basic_ostream<CTy,CTr>& operator<<(std::basic_ostream<CTy,CTr>&, tmat2x4<T,P> const&);
	template <typename CTy, typename CTr, typename T, precision P>
	GLM_FUNC_DECL std::basic_ostream<CTy,CTr>& operator<<(std::basic_ostream<CTy,CTr>&, tmat3x2<T,P> const&);
	template <typename CTy, typename CTr, typename T, precision P>
	GLM_FUNC_DECL std::basic_ostream<CTy,CTr>& operator<<(std::basic_ostream<CTy,CTr>&, tmat3x3<T,P> const&);
	template <typename CTy, typename CTr, typename T, precision P>
	GLM_FUNC_DECL std::basic_ostream<CTy,CTr>& operator<<(std::basic_ostream<CTy,CTr>&, tmat3x4<T,P> const&);
	template <typename CTy, typename CTr, typename T, precision P>
	GLM_FUNC_DECL std::basic_ostream<CTy,CTr>& operator<<(std::basic_ostream<CTy,CTr>&, tmat4x2<T,P> const&);
	template <typename CTy, typename CTr, typename T, precision P>
	GLM_FUNC_DECL std::basic_ostream<CTy,CTr>& operator<<(std::basic_ostream<CTy,CTr>&, tmat4x3<T,P> const&);
	template <typename CTy, typename CTr, typename T, precision P>
	GLM_FUNC_DECL std::basic_ostream<CTy,CTr>& operator<<(std::basic_ostream<CTy,CTr>&, tmat4x4<T,P> const&);

	template <typename CTy, typename CTr, typename T, precision P>
	GLM_FUNC_DECL std::basic_ostream<CTy,CTr> & operator<<(
		std::basic_ostream<CTy,CTr> &,
		std::pair<tmat4x4<T,P> const,
		tmat4x4<T,P> const> const &);

	/// @}
}//namespace glm

#include "io.inl"
