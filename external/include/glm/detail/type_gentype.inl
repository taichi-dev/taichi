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
/// @file glm/detail/type_gentype.inl
/// @date 2008-10-05 / 2011-06-15
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

namespace glm{
namespace detail{

/////////////////////////////////
// Static functions

template <typename vT, uint cT, uint rT, profile pT>
typename base<vT, cT, rT, pT>::size_type base<vT, cT, rT, pT>::col_size()
{
	return cT;
}

template <typename vT, uint cT, uint rT, profile pT>
typename base<vT, cT, rT, pT>::size_type base<vT, cT, rT, pT>::row_size()
{
	return rT;
}

template <typename vT, uint cT, uint rT, profile pT>
typename base<vT, cT, rT, pT>::size_type base<vT, cT, rT, pT>::value_size()
{
	return rT * cT;
}

template <typename vT, uint cT, uint rT, profile pT>
bool base<vT, cT, rT, pT>::is_scalar()
{
	return rT == 1 && cT == 1;
}

template <typename vT, uint cT, uint rT, profile pT>
bool base<vT, cT, rT, pT>::is_vector()
{
	return rT == 1;
}

template <typename vT, uint cT, uint rT, profile pT>
bool base<vT, cT, rT, pT>::is_matrix()
{
	return rT != 1;
}

/////////////////////////////////
// Constructor

template <typename vT, uint cT, uint rT, profile pT>
base<vT, cT, rT, pT>::base()
{
	memset(&this->value, 0, cT * rT * sizeof(vT));
}

template <typename vT, uint cT, uint rT, profile pT>
base<vT, cT, rT, pT>::base
(
	typename base<vT, cT, rT, pT>::class_type const & m
)
{
	for
	(
		typename genType<vT, cT, rT, pT>::size_type i = typename base<vT, cT, rT, pT>::size_type(0);
		i < base<vT, cT, rT, pT>::col_size();
		++i
	)
	{
		this->value[i] = m[i];
	}
}

template <typename vT, uint cT, uint rT, profile pT>
base<vT, cT, rT, pT>::base
(
	typename base<vT, cT, rT, pT>::T const & x
)
{
	if(rT == 1) // vector
	{
		for
		(
			typename base<vT, cT, rT, pT>::size_type i = typename base<vT, cT, rT, pT>::size_type(0);
			i < base<vT, cT, rT, pT>::col_size();
			++i
		)
		{
			this->value[i][rT] = x;
		}
	}
	else // matrix
	{
		memset(&this->value, 0, cT * rT * sizeof(vT));

		typename base<vT, cT, rT, pT>::size_type stop = cT < rT ? cT : rT;

		for
		(
			typename base<vT, cT, rT, pT>::size_type i = typename base<vT, cT, rT, pT>::size_type(0);
			i < stop;
			++i
		)
		{
			this->value[i][i] = x;
		}
	}
}

template <typename vT, uint cT, uint rT, profile pT>
base<vT, cT, rT, pT>::base
(
	typename base<vT, cT, rT, pT>::value_type const * const x
)
{
	memcpy(&this->value, &x.value, cT * rT * sizeof(vT));
}

template <typename vT, uint cT, uint rT, profile pT>
base<vT, cT, rT, pT>::base
(
	typename base<vT, cT, rT, pT>::col_type const * const x
)
{
	for
	(
		typename base<vT, cT, rT, pT>::size_type i = typename base<vT, cT, rT, pT>::size_type(0);
		i < base<vT, cT, rT, pT>::col_size();
		++i
	)
	{
		this->value[i] = x[i];
	}
}

template <typename vT, uint cT, uint rT, profile pT>
template <typename vU, uint cU, uint rU, profile pU>
base<vT, cT, rT, pT>::base
(
	base<vU, cU, rU, pU> const & m
)
{
	for
	(
		typename base<vT, cT, rT, pT>::size_type i = typename base<vT, cT, rT, pT>::size_type(0);
		i < base<vT, cT, rT, pT>::col_size();
		++i
	)
	{
		this->value[i] = base<vT, cT, rT, pT>(m[i]);
	}
}

//////////////////////////////////////
// Accesses

template <typename vT, uint cT, uint rT, profile pT>
typename base<vT, cT, rT, pT>::col_type& base<vT, cT, rT, pT>::operator[]
(
	typename base<vT, cT, rT, pT>::size_type i
)
{
	return this->value[i];
}

template <typename vT, uint cT, uint rT, profile pT>
typename base<vT, cT, rT, pT>::col_type const & base<vT, cT, rT, pT>::operator[]
(
	typename base<vT, cT, rT, pT>::size_type i
) const
{
	return this->value[i];
}

//////////////////////////////////////
// Unary updatable operators

template <typename vT, uint cT, uint rT, profile pT>
typename base<vT, cT, rT, pT>::class_type& base<vT, cT, rT, pT>::operator= 
(
	typename base<vT, cT, rT, pT>::class_type const & x
)
{
	memcpy(&this->value, &x.value, cT * rT * sizeof(vT));
	return *this;
}

template <typename vT, uint cT, uint rT, profile pT>
typename base<vT, cT, rT, pT>::class_type& base<vT, cT, rT, pT>::operator+= 
(
	typename base<vT, cT, rT, pT>::T const & x
)
{
	typename base<vT, cT, rT, pT>::size_type stop_col = x.col_size();
	typename base<vT, cT, rT, pT>::size_type stop_row = x.row_size();

	for(typename base<vT, cT, rT, pT>::size_type j = 0; j < stop_col; ++j)
	for(typename base<vT, cT, rT, pT>::size_type i = 0; i < stop_row; ++i)
		this->value[j][i] += x;

	return *this;
}

template <typename vT, uint cT, uint rT, profile pT>
typename base<vT, cT, rT, pT>::class_type& base<vT, cT, rT, pT>::operator+= 
(
	typename base<vT, cT, rT, pT>::class_type const & x
)
{
	typename base<vT, cT, rT, pT>::size_type stop_col = x.col_size();
	typename base<vT, cT, rT, pT>::size_type stop_row = x.row_size();

	for(typename base<vT, cT, rT, pT>::size_type j = 0; j < stop_col; ++j)
	for(typename base<vT, cT, rT, pT>::size_type i = 0; i < stop_row; ++i)
		this->value[j][i] += x[j][i];

	return *this;
}

template <typename vT, uint cT, uint rT, profile pT>
typename base<vT, cT, rT, pT>::class_type& base<vT, cT, rT, pT>::operator-= 
(
	typename base<vT, cT, rT, pT>::T const & x
)
{
	typename base<vT, cT, rT, pT>::size_type stop_col = x.col_size();
	typename base<vT, cT, rT, pT>::size_type stop_row = x.row_size();

	for(typename base<vT, cT, rT, pT>::size_type j = 0; j < stop_col; ++j)
	for(typename base<vT, cT, rT, pT>::size_type i = 0; i < stop_row; ++i)
		this->value[j][i] -= x;

	return *this;
}

template <typename vT, uint cT, uint rT, profile pT>
typename base<vT, cT, rT, pT>::class_type& base<vT, cT, rT, pT>::operator-= 
(
	typename base<vT, cT, rT, pT>::class_type const & x
)
{
	typename base<vT, cT, rT, pT>::size_type stop_col = x.col_size();
	typename base<vT, cT, rT, pT>::size_type stop_row = x.row_size();

	for(typename base<vT, cT, rT, pT>::size_type j = 0; j < stop_col; ++j)
	for(typename base<vT, cT, rT, pT>::size_type i = 0; i < stop_row; ++i)
		this->value[j][i] -= x[j][i];

	return *this;
}

template <typename vT, uint cT, uint rT, profile pT>
typename base<vT, cT, rT, pT>::class_type& base<vT, cT, rT, pT>::operator*= 
(
	typename base<vT, cT, rT, pT>::T const & x
)
{
	typename base<vT, cT, rT, pT>::size_type stop_col = x.col_size();
	typename base<vT, cT, rT, pT>::size_type stop_row = x.row_size();

	for(typename base<vT, cT, rT, pT>::size_type j = 0; j < stop_col; ++j)
	for(typename base<vT, cT, rT, pT>::size_type i = 0; i < stop_row; ++i)
		this->value[j][i] *= x;

	return *this;
}

template <typename vT, uint cT, uint rT, profile pT>
typename base<vT, cT, rT, pT>::class_type& base<vT, cT, rT, pT>::operator*= 
(
	typename base<vT, cT, rT, pT>::class_type const & x
)
{
	typename base<vT, cT, rT, pT>::size_type stop_col = x.col_size();
	typename base<vT, cT, rT, pT>::size_type stop_row = x.row_size();

	for(typename base<vT, cT, rT, pT>::size_type j = 0; j < stop_col; ++j)
	for(typename base<vT, cT, rT, pT>::size_type i = 0; i < stop_row; ++i)
		this->value[j][i] *= x[j][i];

	return *this;
}

template <typename vT, uint cT, uint rT, profile pT>
typename base<vT, cT, rT, pT>::class_type& base<vT, cT, rT, pT>::operator/= 
(
	typename base<vT, cT, rT, pT>::T const & x
)
{
	typename base<vT, cT, rT, pT>::size_type stop_col = x.col_size();
	typename base<vT, cT, rT, pT>::size_type stop_row = x.row_size();

	for(typename base<vT, cT, rT, pT>::size_type j = 0; j < stop_col; ++j)
	for(typename base<vT, cT, rT, pT>::size_type i = 0; i < stop_row; ++i)
		this->value[j][i] /= x;

	return *this;
}

template <typename vT, uint cT, uint rT, profile pT>
typename base<vT, cT, rT, pT>::class_type& base<vT, cT, rT, pT>::operator/= 
(
	typename base<vT, cT, rT, pT>::class_type const & x
)
{
	typename base<vT, cT, rT, pT>::size_type stop_col = x.col_size();
	typename base<vT, cT, rT, pT>::size_type stop_row = x.row_size();

	for(typename base<vT, cT, rT, pT>::size_type j = 0; j < stop_col; ++j)
	for(typename base<vT, cT, rT, pT>::size_type i = 0; i < stop_row; ++i)
		this->value[j][i] /= x[j][i];

	return *this;
}

template <typename vT, uint cT, uint rT, profile pT>
typename base<vT, cT, rT, pT>::class_type& base<vT, cT, rT, pT>::operator++ ()
{
	typename base<vT, cT, rT, pT>::size_type stop_col = col_size();
	typename base<vT, cT, rT, pT>::size_type stop_row = row_size();

	for(typename base<vT, cT, rT, pT>::size_type j = 0; j < stop_col; ++j)
	for(typename base<vT, cT, rT, pT>::size_type i = 0; i < stop_row; ++i)
		++this->value[j][i];

	return *this;
}

template <typename vT, uint cT, uint rT, profile pT>
typename base<vT, cT, rT, pT>::class_type& base<vT, cT, rT, pT>::operator-- ()
{
	typename base<vT, cT, rT, pT>::size_type stop_col = col_size();
	typename base<vT, cT, rT, pT>::size_type stop_row = row_size();

	for(typename base<vT, cT, rT, pT>::size_type j = 0; j < stop_col; ++j)
	for(typename base<vT, cT, rT, pT>::size_type i = 0; i < stop_row; ++i)
		--this->value[j][i];

	return *this;
}

} //namespace detail
} //namespace glm
