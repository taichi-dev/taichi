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
/// @file glm/gtx/io.inl
/// @date 2013-11-22 / 2014-11-25
/// @author Jan P Springer (regnirpsj@gmail.com)
///////////////////////////////////////////////////////////////////////////////////////////////////

#include <iomanip> // std::setfill<>, std::fixed, std::setprecision, std::right, std::setw
#include <ostream> // std::basic_ostream<>

namespace glm{
namespace io
{
	template <typename CTy>
	/* explicit */ GLM_FUNC_QUALIFIER
	format_punct<CTy>::format_punct(size_t a)
		: std::locale::facet(a),
		formatted         (true),
		precision         (3),
		width             (1 + 4 + 1 + precision),
		separator         (','),
		delim_left        ('['),
		delim_right       (']'),
		space             (' '),
		newline           ('\n'),
		order             (row_major)
	{}

	template <typename CTy>
	/* explicit */ GLM_FUNC_QUALIFIER
	format_punct<CTy>::format_punct(format_punct const& a)
		: std::locale::facet(0),
		formatted         (a.formatted),
		precision         (a.precision),
		width             (a.width),
		separator         (a.separator),
		delim_left        (a.delim_left),
		delim_right       (a.delim_right),
		space             (a.space),
		newline           (a.newline),
		order             (a.order)
	{}

	template <typename CTy> std::locale::id format_punct<CTy>::id;

	template <typename CTy, typename CTr>
	/* explicit */ GLM_FUNC_QUALIFIER basic_state_saver<CTy,CTr>::basic_state_saver(std::basic_ios<CTy,CTr>& a)
		: state_    (a),
		flags_    (a.flags()),
		precision_(a.precision()),
		width_    (a.width()),
		fill_     (a.fill()),
		locale_   (a.getloc())
	{}

	template <typename CTy, typename CTr>
	GLM_FUNC_QUALIFIER basic_state_saver<CTy,CTr>::~basic_state_saver()
	{
		state_.imbue(locale_);
		state_.fill(fill_);
		state_.width(width_);
		state_.precision(precision_);
		state_.flags(flags_);
	}

	template <typename CTy, typename CTr>
	/* explicit */ GLM_FUNC_QUALIFIER basic_format_saver<CTy,CTr>::basic_format_saver(std::basic_ios<CTy,CTr>& a)
		: bss_(a)
	{
		a.imbue(std::locale(a.getloc(), new format_punct<CTy>(get_facet<format_punct<CTy> >(a))));
	}

	template <typename CTy, typename CTr>
	GLM_FUNC_QUALIFIER
	basic_format_saver<CTy,CTr>::~basic_format_saver()
	{}

	/* explicit */ GLM_FUNC_QUALIFIER precision::precision(unsigned a)
		: value(a)
	{}

	/* explicit */ GLM_FUNC_QUALIFIER width::width(unsigned a)
		: value(a)
	{}

	template <typename CTy>
	/* explicit */ GLM_FUNC_QUALIFIER delimeter<CTy>::delimeter(CTy a, CTy b, CTy c)
		: value()
	{
		value[0] = a;
		value[1] = b;
		value[2] = c;
	}

	/* explicit */ GLM_FUNC_QUALIFIER
	order::order(order_type a)
		: value(a)
	{}

	template <typename FTy, typename CTy, typename CTr>
	GLM_FUNC_QUALIFIER FTy const& get_facet(std::basic_ios<CTy,CTr>& ios)
	{
		if (!std::has_facet<FTy>(ios.getloc())) {
		ios.imbue(std::locale(ios.getloc(), new FTy));
		}

		return std::use_facet<FTy>(ios.getloc());
	}

	template <typename CTy, typename CTr>
	GLM_FUNC_QUALIFIER std::basic_ios<CTy,CTr>& formatted(std::basic_ios<CTy,CTr>& ios)
	{
		const_cast<format_punct<CTy>&>(get_facet<format_punct<CTy> >(ios)).formatted = true;

		return ios;
	}

	template <typename CTy, typename CTr>
	GLM_FUNC_QUALIFIER std::basic_ios<CTy,CTr>& unformatted(std::basic_ios<CTy,CTr>& ios)
	{
		const_cast<format_punct<CTy>&>(get_facet<format_punct<CTy> >(ios)).formatted = false;

		return ios;
	}

	template <typename CTy, typename CTr>
	GLM_FUNC_QUALIFIER std::basic_ostream<CTy, CTr>& operator<<(std::basic_ostream<CTy, CTr>& os, precision const& a)
	{
		const_cast<format_punct<CTy>&>(get_facet<format_punct<CTy> >(os)).precision = a.value;

		return os;
	}

	template <typename CTy, typename CTr>
	GLM_FUNC_QUALIFIER std::basic_ostream<CTy, CTr>& operator<<(std::basic_ostream<CTy, CTr>& os, width const& a)
	{
		const_cast<format_punct<CTy>&>(get_facet<format_punct<CTy> >(os)).width = a.value;

		return os;
	}

	template <typename CTy, typename CTr>
	std::basic_ostream<CTy, CTr>& operator<<(std::basic_ostream<CTy, CTr>& os, delimeter<CTy> const& a)
	{
		format_punct<CTy> & fmt(const_cast<format_punct<CTy>&>(get_facet<format_punct<CTy> >(os)));

		fmt.delim_left  = a.value[0];
		fmt.delim_right = a.value[1];
		fmt.separator   = a.value[2];

		return os;
	}

	template <typename CTy, typename CTr>
	GLM_FUNC_QUALIFIER std::basic_ostream<CTy, CTr>& operator<<(std::basic_ostream<CTy, CTr>& os, order const& a)
	{
		const_cast<format_punct<CTy>&>(get_facet<format_punct<CTy> >(os)).order = a.value;

		return os;
	}
} // namespace io

	template <typename CTy, typename CTr, typename T, precision P>
	GLM_FUNC_QUALIFIER std::basic_ostream<CTy,CTr>& operator<<(std::basic_ostream<CTy,CTr>& os, tquat<T,P> const& a)
	{
		typename std::basic_ostream<CTy,CTr>::sentry const cerberus(os);

		if(cerberus)
		{
			io::format_punct<CTy> const & fmt(io::get_facet<io::format_punct<CTy> >(os));

			if(fmt.formatted)
			{
				io::basic_state_saver<CTy> const bss(os);

				os << std::fixed
					<< std::right
					<< std::setprecision(fmt.precision)
					<< std::setfill(fmt.space)
					<< fmt.delim_left
					<< std::setw(fmt.width) << a.w << fmt.separator
					<< std::setw(fmt.width) << a.x << fmt.separator
					<< std::setw(fmt.width) << a.y << fmt.separator
					<< std::setw(fmt.width) << a.z
					<< fmt.delim_right;
			}
			else
			{
				os << a.w << fmt.space << a.x << fmt.space << a.y << fmt.space << a.z;
			}
		}

		return os;
	}

	template <typename CTy, typename CTr, typename T, precision P>
	GLM_FUNC_QUALIFIER std::basic_ostream<CTy,CTr>& operator<<(std::basic_ostream<CTy,CTr>& os, tvec2<T,P> const& a)
	{
		typename std::basic_ostream<CTy,CTr>::sentry const cerberus(os);

		if(cerberus)
		{
			io::format_punct<CTy> const & fmt(io::get_facet<io::format_punct<CTy> >(os));

			if(fmt.formatted)
			{
				io::basic_state_saver<CTy> const bss(os);

				os << std::fixed
					<< std::right
					<< std::setprecision(fmt.precision)
					<< std::setfill(fmt.space)
					<< fmt.delim_left
					<< std::setw(fmt.width) << a.x << fmt.separator
					<< std::setw(fmt.width) << a.y
					<< fmt.delim_right;
			}
			else
			{
				os << a.x << fmt.space << a.y;
			}
		}

		return os;
	}

	template <typename CTy, typename CTr, typename T, precision P>
	GLM_FUNC_QUALIFIER std::basic_ostream<CTy,CTr>& operator<<(std::basic_ostream<CTy,CTr>& os, tvec3<T,P> const& a)
	{
		typename std::basic_ostream<CTy,CTr>::sentry const cerberus(os);

		if(cerberus)
		{
			io::format_punct<CTy> const & fmt(io::get_facet<io::format_punct<CTy> >(os));

			if(fmt.formatted)
			{
				io::basic_state_saver<CTy> const bss(os);

				os << std::fixed
					<< std::right
					<< std::setprecision(fmt.precision)
					<< std::setfill(fmt.space)
					<< fmt.delim_left
					<< std::setw(fmt.width) << a.x << fmt.separator
					<< std::setw(fmt.width) << a.y << fmt.separator
					<< std::setw(fmt.width) << a.z
					<< fmt.delim_right;
			}
			else
			{
				os << a.x << fmt.space << a.y << fmt.space << a.z;
			}
		}

		return os;
	}

	template <typename CTy, typename CTr, typename T, precision P>
	GLM_FUNC_QUALIFIER std::basic_ostream<CTy,CTr>& operator<<(std::basic_ostream<CTy,CTr>& os, tvec4<T,P> const& a)
	{
		typename std::basic_ostream<CTy,CTr>::sentry const cerberus(os);

		if(cerberus)
		{
			io::format_punct<CTy> const & fmt(io::get_facet<io::format_punct<CTy> >(os));

			if(fmt.formatted)
			{
				io::basic_state_saver<CTy> const bss(os);

				os << std::fixed
					<< std::right
					<< std::setprecision(fmt.precision)
					<< std::setfill(fmt.space)
					<< fmt.delim_left
					<< std::setw(fmt.width) << a.x << fmt.separator
					<< std::setw(fmt.width) << a.y << fmt.separator
					<< std::setw(fmt.width) << a.z << fmt.separator
					<< std::setw(fmt.width) << a.w
					<< fmt.delim_right;
			}
			else
			{
				os << a.x << fmt.space << a.y << fmt.space << a.z << fmt.space << a.w;
			}
		}

		return os;
	}

	template <typename CTy, typename CTr, typename T, precision P>
	GLM_FUNC_QUALIFIER std::basic_ostream<CTy,CTr>& operator<<(std::basic_ostream<CTy,CTr>& os, tmat2x2<T,P> const& a)
	{
		typename std::basic_ostream<CTy,CTr>::sentry const cerberus(os);

		if(cerberus)
		{
			io::format_punct<CTy> const & fmt(io::get_facet<io::format_punct<CTy> >(os));
			tmat2x2<T,P> m(a);

			if(io::row_major == fmt.order)
				m = transpose(a);

			if(fmt.formatted)
			{
				os << fmt.newline
					<< fmt.delim_left << m[0] << fmt.newline
					<< fmt.space      << m[1] << fmt.delim_right;
			}
			else
			{
				os << m[0] << fmt.space << m[1];
			}
		}

		return os;
	}

	template <typename CTy, typename CTr, typename T, precision P>
	GLM_FUNC_QUALIFIER std::basic_ostream<CTy,CTr>& operator<<(std::basic_ostream<CTy,CTr>& os, tmat2x3<T,P> const& a)
	{
		typename std::basic_ostream<CTy,CTr>::sentry const cerberus(os);

		if(cerberus)
		{
			io::format_punct<CTy> const & fmt(io::get_facet<io::format_punct<CTy> >(os));
			tmat3x2<T,P> m(a);

			if(io::row_major == fmt.order)
				m = transpose(a);

			if(fmt.formatted)
			{
				os << fmt.newline
					<< fmt.delim_left << m[0] << fmt.newline
					<< fmt.space      << m[1] << fmt.newline
					<< fmt.space      << m[2] << fmt.delim_right;
			}
			else
			{
				os << m[0] << fmt.space << m[1] << fmt.space << m[2];
			}
		}

		return os;
	}

	template <typename CTy, typename CTr, typename T, precision P>
	GLM_FUNC_QUALIFIER std::basic_ostream<CTy,CTr>& operator<<(std::basic_ostream<CTy,CTr>& os, tmat2x4<T,P> const& a)
	{
		typename std::basic_ostream<CTy,CTr>::sentry const cerberus(os);

		if(cerberus)
		{
			io::format_punct<CTy> const & fmt(io::get_facet<io::format_punct<CTy> >(os));
			tmat4x2<T,P> m(a);

			if(io::row_major == fmt.order)
				m = transpose(a);


			if(fmt.formatted)
			{
				os << fmt.newline
					<< fmt.delim_left << m[0] << fmt.newline
					<< fmt.space      << m[1] << fmt.newline
					<< fmt.space      << m[2] << fmt.newline
					<< fmt.space      << m[3] << fmt.delim_right;
			}
			else
			{
				os << m[0] << fmt.space << m[1] << fmt.space << m[2] << fmt.space << m[3];
			}
		}

		return os;
	}

	template <typename CTy, typename CTr, typename T, precision P>
	GLM_FUNC_QUALIFIER std::basic_ostream<CTy,CTr>& operator<<(std::basic_ostream<CTy,CTr>& os, tmat3x2<T,P> const& a)
	{
		typename std::basic_ostream<CTy,CTr>::sentry const cerberus(os);

		if(cerberus)
		{
			io::format_punct<CTy> const & fmt(io::get_facet<io::format_punct<CTy> >(os));
			tmat2x3<T,P> m(a);

			if(io::row_major == fmt.order)
				m = transpose(a);

			if(fmt.formatted)
			{
				os << fmt.newline
					<< fmt.delim_left << m[0] << fmt.newline
					<< fmt.space      << m[1] << fmt.delim_right;
			}
			else
			{
				os << m[0] << fmt.space << m[1];
			}
		}

		return os;
	}

	template <typename CTy, typename CTr, typename T, precision P>
	GLM_FUNC_QUALIFIER std::basic_ostream<CTy,CTr>& operator<<(std::basic_ostream<CTy,CTr>& os, tmat3x3<T,P> const& a)
	{
		typename std::basic_ostream<CTy,CTr>::sentry const cerberus(os);

		if(cerberus)
		{
			io::format_punct<CTy> const & fmt(io::get_facet<io::format_punct<CTy> >(os));
			tmat3x3<T,P> m(a);

			if(io::row_major == fmt.order)
				m = transpose(a);

			if(fmt.formatted)
			{
				os << fmt.newline
					<< fmt.delim_left << m[0] << fmt.newline
					<< fmt.space      << m[1] << fmt.newline
					<< fmt.space      << m[2] << fmt.delim_right;
			}
			else
			{
				os << m[0] << fmt.space << m[1] << fmt.space << m[2];
			}
		}

		return os;
	}

	template <typename CTy, typename CTr, typename T, precision P>
	GLM_FUNC_QUALIFIER std::basic_ostream<CTy,CTr> & operator<<(std::basic_ostream<CTy,CTr>& os, tmat3x4<T,P> const& a)
	{
		typename std::basic_ostream<CTy,CTr>::sentry const cerberus(os);

		if(cerberus)
		{
			io::format_punct<CTy> const & fmt(io::get_facet<io::format_punct<CTy> >(os));
			tmat4x3<T,P> m(a);

			if(io::row_major == fmt.order)
				m = transpose(a);

			if (fmt.formatted)
			{
				os << fmt.newline
					<< fmt.delim_left << m[0] << fmt.newline
					<< fmt.space      << m[1] << fmt.newline
					<< fmt.space      << m[2] << fmt.newline
					<< fmt.space      << m[3] << fmt.delim_right;
			}
			else
			{
				os << m[0] << fmt.space << m[1] << fmt.space << m[2] << fmt.space << m[3];
			}
		}

		return os;
	}

	template <typename CTy, typename CTr, typename T, precision P>
	GLM_FUNC_QUALIFIER std::basic_ostream<CTy,CTr> & operator<<(std::basic_ostream<CTy,CTr>& os, tmat4x2<T,P> const& a)
	{
		typename std::basic_ostream<CTy,CTr>::sentry const cerberus(os);

		if(cerberus)
		{
			io::format_punct<CTy> const & fmt(io::get_facet<io::format_punct<CTy> >(os));
			tmat2x4<T,P> m(a);

			if(io::row_major == fmt.order)
				m = transpose(a);

			if (fmt.formatted)
			{
				os << fmt.newline
					<< fmt.delim_left << m[0] << fmt.newline
					<< fmt.space      << m[1] << fmt.delim_right;
			}
			else
			{
				os << m[0] << fmt.space << m[1];
			}
		}

		return os;
	}

	template <typename CTy, typename CTr, typename T, precision P>
	GLM_FUNC_QUALIFIER std::basic_ostream<CTy,CTr> & operator<<(std::basic_ostream<CTy,CTr>& os, tmat4x3<T,P> const& a)
	{
		typename std::basic_ostream<CTy,CTr>::sentry const cerberus(os);

		if(cerberus)
		{
			io::format_punct<CTy> const & fmt(io::get_facet<io::format_punct<CTy> >(os));
			tmat3x4<T,P> m(a);

			if(io::row_major == fmt.order)
				m = transpose(a);

			if(fmt.formatted)
			{
				os << fmt.newline
					<< fmt.delim_left << m[0] << fmt.newline
					<< fmt.space      << m[1] << fmt.newline
					<< fmt.space      << m[2] << fmt.delim_right;
			}
			else
			{
				os << m[0] << fmt.space << m[1] << fmt.space << m[2];
			}
		}

		return os;
	}

	template <typename CTy, typename CTr, typename T, precision P>
	GLM_FUNC_QUALIFIER std::basic_ostream<CTy,CTr> & operator<<(std::basic_ostream<CTy,CTr>& os, tmat4x4<T,P> const& a)
	{
		typename std::basic_ostream<CTy,CTr>::sentry const cerberus(os);

		if(cerberus)
		{
			io::format_punct<CTy> const & fmt(io::get_facet<io::format_punct<CTy> >(os));
			tmat4x4<T,P> m(a);

			if (io::row_major == fmt.order)
				m = transpose(a);

			if(fmt.formatted)
			{
				os << fmt.newline
					<< fmt.delim_left << m[0] << fmt.newline
					<< fmt.space      << m[1] << fmt.newline
					<< fmt.space      << m[2] << fmt.newline
					<< fmt.space      << m[3] << fmt.delim_right;
			}
			else
			{
				os << m[0] << fmt.space << m[1] << fmt.space << m[2] << fmt.space << m[3];
			}
		}

		return os;
	}

	template <typename CTy, typename CTr, typename T, precision P>
	GLM_FUNC_QUALIFIER std::basic_ostream<CTy,CTr>& operator<<(
		std::basic_ostream<CTy,CTr> & os,
		std::pair<tmat4x4<T,P> const, tmat4x4<T,P> const> const& a)
	{
		typename std::basic_ostream<CTy,CTr>::sentry const cerberus(os);

		if(cerberus)
		{
			io::format_punct<CTy> const & fmt(io::get_facet<io::format_punct<CTy> >(os));
			tmat4x4<T,P> ml(a.first);
			tmat4x4<T,P> mr(a.second);

			if(io::row_major == fmt.order)
			{
				ml = transpose(a.first);
				mr = transpose(a.second);
			}

			if(fmt.formatted)
			{
				CTy const & l(fmt.delim_left);
				CTy const & r(fmt.delim_right);
				CTy const & s(fmt.space);

				os << fmt.newline
					<< l << ml[0] << s << s << l << mr[0] << fmt.newline
					<< s << ml[1] << s << s << s << mr[1] << fmt.newline
					<< s << ml[2] << s << s << s << mr[2] << fmt.newline
					<< s << ml[3] << r << s << s << mr[3] << r;
			}
			else
			{
				os << ml << fmt.space << mr;
			}
		}

		return os;
	}
}//namespace glm
