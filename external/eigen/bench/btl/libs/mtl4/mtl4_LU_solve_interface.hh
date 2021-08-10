//=====================================================
// File   :  blitz_LU_solve_interface.hh
// Author :  L. Plagne <laurent.plagne@edf.fr)>        
// Copyright (C) EDF R&D,  lun sep 30 14:23:31 CEST 2002
//=====================================================
// 
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
// 
#ifndef BLITZ_LU_SOLVE_INTERFACE_HH
#define BLITZ_LU_SOLVE_INTERFACE_HH

#include "blitz/array.h"
#include <vector>

BZ_USING_NAMESPACE(blitz)

template<class real>
class blitz_LU_solve_interface : public blitz_interface<real>
{

public :

  typedef typename blitz_interface<real>::gene_matrix gene_matrix;
  typedef typename blitz_interface<real>::gene_vector gene_vector;

  typedef blitz::Array<int,1> Pivot_Vector;

  inline static void new_Pivot_Vector(Pivot_Vector & pivot,int N)
  {

    pivot.resize(N);

  }

  inline static void free_Pivot_Vector(Pivot_Vector & pivot)
  {
    
    return;

  }


  static inline real matrix_vector_product_sliced(const gene_matrix & A, gene_vector B, int row, int col_start, int col_end)
  {
    
    real somme=0.;
    
    for (int j=col_start ; j<col_end+1 ; j++){
	
	somme+=A(row,j)*B(j);
	
    }

    return somme;

  }




  static inline real matrix_matrix_product_sliced(gene_matrix & A, int row, int col_start, int col_end, gene_matrix & B, int row_shift, int col )
  {
    
    real somme=0.;
    
    for (int j=col_start ; j<col_end+1 ; j++){
	
	somme+=A(row,j)*B(j+row_shift,col);
	
    }

    return somme;

  }

  inline static void LU_factor(gene_matrix & LU, Pivot_Vector & pivot, int N)
  {

    ASSERT( LU.rows()==LU.cols() ) ;
    int index_max = 0 ;
    real big = 0. ;
    real theSum = 0. ;
    real dum = 0. ;
    // Get the implicit scaling information :
    gene_vector ImplicitScaling( N ) ;
    for( int i=0; i<N; i++ ) {
      big = 0. ;
      for( int j=0; j<N; j++ ) {
	if( abs( LU( i, j ) )>=big ) big = abs( LU( i, j ) ) ;
      }
      if( big==0. ) {
	INFOS( "blitz_LU_factor::Singular matrix" ) ;
	exit( 0 ) ;
      }
      ImplicitScaling( i ) = 1./big ;
    }
    // Loop over columns of Crout's method :
    for( int j=0; j<N; j++ ) {
      for( int i=0; i<j; i++ ) {
	theSum = LU( i, j ) ;
	theSum -= matrix_matrix_product_sliced(LU, i, 0, i-1, LU, 0, j) ;
	//	theSum -= sum( LU( i, Range( fromStart, i-1 ) )*LU( Range( fromStart, i-1 ), j ) ) ;
	LU( i, j ) = theSum ;
      }
      
      // Search for the largest pivot element :
      big = 0. ;
      for( int i=j; i<N; i++ ) {
	theSum = LU( i, j ) ;
	theSum -= matrix_matrix_product_sliced(LU, i, 0, j-1, LU, 0, j) ;
	//	theSum -= sum( LU( i, Range( fromStart, j-1 ) )*LU( Range( fromStart, j-1 ), j ) ) ;
	LU( i, j ) = theSum ;
	if( (ImplicitScaling( i )*abs( theSum ))>=big ) {
	  dum = ImplicitScaling( i )*abs( theSum ) ;
	  big = dum ;
	  index_max = i ;
	}
      }
      // Interchanging rows and the scale factor :
      if( j!=index_max ) {
	for( int k=0; k<N; k++ ) {
	  dum = LU( index_max, k ) ;
	  LU( index_max, k ) = LU( j, k ) ;
	  LU( j, k ) = dum ;
	}
	ImplicitScaling( index_max ) = ImplicitScaling( j ) ;
      }
      pivot( j ) = index_max ;
      if ( LU( j, j )==0. ) LU( j, j ) = 1.e-20 ;
      // Divide by the pivot element :
      if( j<N ) {
	dum = 1./LU( j, j ) ;
	for( int i=j+1; i<N; i++ ) LU( i, j ) *= dum ;
      }
    }

  }

  inline static void LU_solve(const gene_matrix & LU, const Pivot_Vector pivot, gene_vector &B, gene_vector X, int N)
  {

    // Pour conserver le meme header, on travaille sur X, copie du second-membre B
    X = B.copy() ;
    ASSERT( LU.rows()==LU.cols() ) ;
    firstIndex indI ;
    // Forward substitution :
    int ii = 0 ;
    real theSum = 0. ;
    for( int i=0; i<N; i++ ) {
      int ip = pivot( i ) ;
      theSum = X( ip ) ;
      //      theSum = B( ip ) ;
      X( ip ) = X( i ) ;
      //      B( ip ) = B( i ) ;
      if( ii ) {
	theSum -= matrix_vector_product_sliced(LU, X, i, ii-1, i-1) ;
	//	theSum -= sum( LU( i, Range( ii-1, i-1 ) )*X( Range( ii-1, i-1 ) ) ) ;
	//	theSum -= sum( LU( i, Range( ii-1, i-1 ) )*B( Range( ii-1, i-1 ) ) ) ;
      } else if( theSum ) {
	ii = i+1 ;
      }
      X( i ) = theSum ;
      //      B( i ) = theSum ;
    }
    // Backsubstitution :
    for( int i=N-1; i>=0; i-- ) {
      theSum = X( i ) ;
      //      theSum = B( i ) ;
      theSum -= matrix_vector_product_sliced(LU, X, i, i+1, N) ;
      //      theSum -= sum( LU( i, Range( i+1, toEnd ) )*X( Range( i+1, toEnd ) ) ) ;
      //      theSum -= sum( LU( i, Range( i+1, toEnd ) )*B( Range( i+1, toEnd ) ) ) ;
      // Store a component of the solution vector :
      X( i ) = theSum/LU( i, i ) ;
      //      B( i ) = theSum/LU( i, i ) ;
    }

  }

};

#endif
