//#####################################################################
// Copyright 2013, Sean Bauer, Eftychios Sifakis.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include <SPGrid/Tools/Shadow_Grid_Helper.h>

using namespace SPGrid;

//#####################################################################
// Function Shadow_Grid_Helper (2D)
//#####################################################################
template <class T, int log2_struct>
void Shadow_Grid_Helper<T,log2_struct,2>::ComputeShadowGrid(unsigned long* offset_grid_ptr,unsigned long packed_offset)
{
    typedef unsigned long (&offset_grid_type)[og_xsize][og_ysize];
    offset_grid_type o_grid = reinterpret_cast<offset_grid_type>(*offset_grid_ptr);
    
    unsigned long simple_offset = 0;
    // Fill in simple offsets
    for (int i = xmin; i<=xmax; i++)
    for (int j = ymin; j<=ymax; j++)
    {
      o_grid[i][j] = packed_offset + simple_offset;  // Can do simple addition here since addresses are within block
      simple_offset += sizeof(T);
    }

    //Vertices
    o_grid[xmin-1][ymin-1] = T_MASK::template Packed_Offset<-1,-1>(o_grid[xmin][ymin]);
    o_grid[xmin-1][ymax+1] = T_MASK::template Packed_Offset<-1, 1>(o_grid[xmin][ymax]);
    o_grid[xmax+1][ymin-1] = T_MASK::template Packed_Offset< 1,-1>(o_grid[xmax][ymin]);
    o_grid[xmax+1][ymax+1] = T_MASK::template Packed_Offset< 1, 1>(o_grid[xmax][ymax]);

    // 4 edges
    simple_offset = o_grid[xmin][ymin-1] = T_MASK::template Packed_Offset< 0,-1>(o_grid[xmin][ymin]);
    for( int i = xmin+1; i <= xmax; i++ ) o_grid[i][ymin-1] = simple_offset + (i-1)*(sizeof(T) * (block_ysize));
    simple_offset = o_grid[xmin][ymax+1] = T_MASK::template Packed_Offset< 0, 1>(o_grid[xmin][ymax]);
    for( int i = xmin+1; i <= xmax; i++ ) o_grid[i][ymax+1] = simple_offset + (i-1)*(sizeof(T) * (block_ysize));

    simple_offset = o_grid[xmin-1][ymin] = T_MASK::template Packed_Offset<-1, 0>(o_grid[xmin][ymin]);
    for( int j = ymin+1; j <= ymax; j++ ) o_grid[xmin-1][j] = simple_offset + (j-1)*(sizeof(T));
    simple_offset = o_grid[xmax+1][ymin] = T_MASK::template Packed_Offset< 1, 0>(o_grid[xmax][ymin]);
    for( int j = ymin+1; j <= ymax; j++ ) o_grid[xmax+1][j] = simple_offset + (j-1)*(sizeof(T));

    // First let's do the starting points
    o_grid[xmin-1][ymin] =  T_MASK::template Packed_OffsetXdim<-1>(o_grid[xmin][ymin]);
    o_grid[xmax+1][ymin] =  T_MASK::template Packed_OffsetXdim< 1>(o_grid[xmax][ymin]);

    o_grid[xmin][ymin-1] =  T_MASK::template Packed_OffsetYdim<-1>(o_grid[xmin][ymin]);
    o_grid[xmin][ymax+1] =  T_MASK::template Packed_OffsetYdim< 1>(o_grid[xmin][ymax]);

    // Fill in edge offsets (cube faces, but not edges will be correct after this)
    // This is ok for 6 neighbors, but one more pass will be needed for kernels that use edges
    {
      // Left and Right face
      for (int i=xmin-1; i<=xmax+1; i+= (xmax-xmin)+2)
      {
        simple_offset = o_grid[i][ymin];
        for (int j=ymin; j<=ymax; j++)
        {
          o_grid[i][j] = simple_offset;
          simple_offset += sizeof(T);  // Simple addition (going through neighboring block in same manner)
        }
      }
    }
    
    {
      // Top and bottom face
      for (int j = ymin-1; j<=ymax+1; j+= (ymax-ymin)+2)
      {
        simple_offset = o_grid[xmin][j];
        for (int i=xmin; i<=xmax; i++)
        {
          o_grid[i][j] = simple_offset;
          simple_offset += sizeof(T) * (block_ysize);
        }
      }
    }
}
//#####################################################################
// Function Shadow_Grid_Helper (3D)
//#####################################################################
template <class T, int log2_struct>
void Shadow_Grid_Helper<T,log2_struct,3>::ComputeShadowGrid(unsigned long* offset_grid_ptr,unsigned long packed_offset)
{
    typedef unsigned long (&offset_grid_type)[og_xsize][og_ysize][og_zsize];
    offset_grid_type o_grid = reinterpret_cast<offset_grid_type>(*offset_grid_ptr);

    // clear o_grid
    for(int i=0;i<og_xsize;i++) for(int j=0;j<og_ysize;j++) for(int k=0;k<og_zsize;k++) o_grid[i][j][k]=packed_offset;

    unsigned long simple_offset = 0;
    // Fill in simple offsets
    for (int i = xmin; i<=xmax; i++)
    for (int j = ymin; j<=ymax; j++)
    for (int k = zmin; k<=zmax; k++)
    {
      o_grid[i][j][k] = packed_offset + simple_offset;  // Can do simple addition here since addresses are within block
      simple_offset += sizeof(T);
    }

    // Vertices
    o_grid[xmin-1][ymin-1][zmin-1] = T_MASK::template Packed_Offset<-1,-1,-1>(o_grid[xmin][ymin][zmin]);
    o_grid[xmin-1][ymin-1][zmax+1] = T_MASK::template Packed_Offset<-1,-1, 1>(o_grid[xmin][ymin][zmax]);
    o_grid[xmin-1][ymax+1][zmin-1] = T_MASK::template Packed_Offset<-1, 1,-1>(o_grid[xmin][ymax][zmin]);
    o_grid[xmin-1][ymax+1][zmax+1] = T_MASK::template Packed_Offset<-1, 1, 1>(o_grid[xmin][ymax][zmax]);
    o_grid[xmax+1][ymin-1][zmin-1] = T_MASK::template Packed_Offset< 1,-1,-1>(o_grid[xmax][ymin][zmin]);
    o_grid[xmax+1][ymin-1][zmax+1] = T_MASK::template Packed_Offset< 1,-1, 1>(o_grid[xmax][ymin][zmax]);
    o_grid[xmax+1][ymax+1][zmin-1] = T_MASK::template Packed_Offset< 1, 1,-1>(o_grid[xmax][ymax][zmin]);
    o_grid[xmax+1][ymax+1][zmax+1] = T_MASK::template Packed_Offset< 1, 1, 1>(o_grid[xmax][ymax][zmax]);

    // 12 edges
    simple_offset = o_grid[xmin][ymin-1][zmin-1] = T_MASK::template Packed_Offset< 0,-1,-1>(o_grid[xmin][ymin][zmin]);
    for( int i = xmin+1; i <= xmax; i++ ) o_grid[i][ymin-1][zmin-1] = simple_offset + (i-1)*(sizeof(T) * (block_ysize) * (block_zsize));
    simple_offset = o_grid[xmin][ymin-1][zmax+1] = T_MASK::template Packed_Offset< 0,-1, 1>(o_grid[xmin][ymin][zmax]);
    for( int i = xmin+1; i <= xmax; i++ ) o_grid[i][ymin-1][zmax+1] = simple_offset + (i-1)*(sizeof(T) * (block_ysize) * (block_zsize));
    simple_offset = o_grid[xmin][ymax+1][zmin-1] = T_MASK::template Packed_Offset< 0, 1,-1>(o_grid[xmin][ymax][zmin]);
    for( int i = xmin+1; i <= xmax; i++ ) o_grid[i][ymax+1][zmin-1] = simple_offset + (i-1)*(sizeof(T) * (block_ysize) * (block_zsize));
    simple_offset = o_grid[xmin][ymax+1][zmax+1] = T_MASK::template Packed_Offset< 0, 1, 1>(o_grid[xmin][ymax][zmax]);
    for( int i = xmin+1; i <= xmax; i++ ) o_grid[i][ymax+1][zmax+1] = simple_offset + (i-1)*(sizeof(T) * (block_ysize) * (block_zsize));
    simple_offset = o_grid[xmin-1][ymin][zmin-1] = T_MASK::template Packed_Offset<-1, 0,-1>(o_grid[xmin][ymin][zmin]);
    for( int j = ymin+1; j <= ymax; j++ ) o_grid[xmin-1][j][zmin-1] = simple_offset + (j-1)*(block_zsize*sizeof(T));
    simple_offset = o_grid[xmin-1][ymin][zmax+1] = T_MASK::template Packed_Offset<-1, 0, 1>(o_grid[xmin][ymin][zmax]);
    for( int j = ymin+1; j <= ymax; j++ ) o_grid[xmin-1][j][zmax+1] = simple_offset + (j-1)*(block_zsize*sizeof(T));
    simple_offset = o_grid[xmax+1][ymin][zmin-1] = T_MASK::template Packed_Offset< 1, 0,-1>(o_grid[xmax][ymin][zmin]);
    for( int j = ymin+1; j <= ymax; j++ ) o_grid[xmax+1][j][zmin-1] = simple_offset + (j-1)*(block_zsize*sizeof(T));
    simple_offset = o_grid[xmax+1][ymin][zmax+1] = T_MASK::template Packed_Offset< 1, 0, 1>(o_grid[xmax][ymin][zmax]);
    for( int j = ymin+1; j <= ymax; j++ ) o_grid[xmax+1][j][zmax+1] = simple_offset + (j-1)*(block_zsize*sizeof(T));
    simple_offset = o_grid[xmin-1][ymin-1][zmin] = T_MASK::template Packed_Offset<-1,-1, 0>(o_grid[xmin][ymin][zmin]);
    for( int k = zmin+1; k <= zmax; k++ ) o_grid[xmin-1][ymin-1][k] = simple_offset + (k-1)*sizeof(T);
    simple_offset = o_grid[xmin-1][ymax+1][zmin] = T_MASK::template Packed_Offset<-1, 1, 0>(o_grid[xmin][ymax][zmin]);
    for( int k = zmin+1; k <= zmax; k++ ) o_grid[xmin-1][ymax+1][k] = simple_offset + (k-1)*sizeof(T);
    simple_offset = o_grid[xmax+1][ymin-1][zmin] = T_MASK::template Packed_Offset< 1,-1, 0>(o_grid[xmax][ymin][zmin]);
    for( int k = zmin+1; k <= zmax; k++ ) o_grid[xmax+1][ymin-1][k] = simple_offset + (k-1)*sizeof(T);
    simple_offset = o_grid[xmax+1][ymax+1][zmin] = T_MASK::template Packed_Offset< 1, 1, 0>(o_grid[xmax][ymax][zmin]);
    for( int k = zmin+1; k <= zmax; k++ ) o_grid[xmax+1][ymax+1][k] = simple_offset + (k-1)*sizeof(T);

    // First let's do the starting points
    o_grid[xmin-1][ymin][zmin] =  T_MASK::template Packed_OffsetXdim<-1>(o_grid[xmin][ymin][zmin]);
    o_grid[xmax+1][ymin][zmin] =  T_MASK::template Packed_OffsetXdim< 1>(o_grid[xmax][ymin][zmin]);

    o_grid[xmin][ymin][zmin-1] =  T_MASK::template Packed_OffsetZdim<-1>(o_grid[xmin][ymin][zmin]);
    o_grid[xmin][ymin][zmax+1] =  T_MASK::template Packed_OffsetZdim< 1>(o_grid[xmin][ymin][zmax]);

    o_grid[xmin][ymin-1][zmin] =  T_MASK::template Packed_OffsetYdim<-1>(o_grid[xmin][ymin][zmin]);
    o_grid[xmin][ymax+1][zmin] =  T_MASK::template Packed_OffsetYdim< 1>(o_grid[xmin][ymax][zmin]);

    // Fill in edge offsets (cube faces, but not edges will be correct after this)
    // This is ok for 6 neighbors, but one more pass will be needed for kernels that use edges
    {
      // Left and Right face
      for (int i=xmin-1; i<=xmax+1; i+= (xmax-xmin)+2)
      {
        simple_offset = o_grid[i][ymin][zmin];
        for (int j=ymin; j<=ymax; j++)
        for (int k=zmin; k<=zmax; k++)
        {
          o_grid[i][j][k] = simple_offset;
          simple_offset += sizeof(T);  // Simple addition (going through neighboring block in same manner)
        }
      }
    }

    {
      // Front and Back face
      for (int k=zmin-1; k<=zmax+1; k+= (zmax-zmin)+2)
      {
        simple_offset = o_grid[xmin][ymin][k];
        for (int i=xmin; i<=xmax; i++)
        for (int j=ymin; j<=ymax; j++)
        {
          o_grid[i][j][k] = simple_offset;
          simple_offset += block_zsize*sizeof(T);  
        }
      }
    }
    
    {
      // Top and bottom face
      for (int j=ymin-1; j<=ymax+1; j+= (ymax-ymin)+2)
      {
        simple_offset = o_grid[xmin][j][zmin];
        for (int i=xmin; i<=xmax; i++)
        {
          for (int k=zmin; k<=zmax; k++)
          {
            o_grid[i][j][k] = simple_offset;
            simple_offset += sizeof(T);  
          }
          simple_offset += sizeof(T) * (block_ysize-1) * (block_zsize);
        }
      }
    }
}
//#####################################################################
namespace SPGrid {
template
class Shadow_Grid_Helper<float, 5, 2>;

template
class Shadow_Grid_Helper<float, 6, 2>;

template
class Shadow_Grid_Helper<float, 5, 3>;

template
class Shadow_Grid_Helper<float, 6, 3>;

#ifndef COMPILE_WITHOUT_DOUBLE_SUPPORT

template
class Shadow_Grid_Helper<double, 5, 2>;

template
class Shadow_Grid_Helper<double, 6, 2>;

template
class Shadow_Grid_Helper<double, 5, 3>;

template
class Shadow_Grid_Helper<double, 6, 3>;

#endif
}
