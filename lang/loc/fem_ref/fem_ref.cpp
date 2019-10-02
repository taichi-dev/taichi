template <int Ix,
          int Iy,
          int Iz,  // output
          int Jx,
          int Jy,
          int Jz>  // input
__forceinline static void
Apply_Spoke(uint64_t base_offset,
            SIMD_Type f[3],
            Const_Array_Type u_x,
            Const_Array_Type u_y,
            Const_Array_Type u_z,
            Const_Array_Type mu,
            Const_Array_Type lambda,
            const T (&K_mu_array)[8][8][3][3],
            const T (&K_la_array)[8][8][3][3]) {
  constexpr int cell_x = -Ix;
  constexpr int cell_y = -Iy;
  constexpr int cell_z = -Iz;
  constexpr int node_x = Jx - Ix;
  constexpr int node_y = Jy - Iy;
  constexpr int node_z = Jz - Iz;
  constexpr int index_in = Jx * 4 + Jy * 2 + Jz;
  constexpr int index_ou = Ix * 4 + Iy * 2 + Iz;
  SIMD_Type Vu_x, Vu_y, Vu_z;
  SIMD_Utilities::template Get_Vector<node_x, node_y, node_z>(
      base_offset, u_x, u_y, u_z, Vu_x, Vu_y, Vu_z);
  SIMD_Type Vmu, Vla;
  SIMD_Utilities::template Get_Vector<cell_x, cell_y, cell_z>(base_offset, mu,
                                                              lambda, Vmu, Vla);
  SIMD_Type K_xx_mu =
      SIMD_Operations::mul(Vmu, K_mu_array[index_ou][index_in][0][0]);
  SIMD_Type K_xy_mu =
      SIMD_Operations::mul(Vmu, K_mu_array[index_ou][index_in][0][1]);
  SIMD_Type K_xz_mu =
      SIMD_Operations::mul(Vmu, K_mu_array[index_ou][index_in][0][2]);
  SIMD_Type K_yx_mu =
      SIMD_Operations::mul(Vmu, K_mu_array[index_ou][index_in][1][0]);
  SIMD_Type K_yy_mu =
      SIMD_Operations::mul(Vmu, K_mu_array[index_ou][index_in][1][1]);
  SIMD_Type K_yz_mu =
      SIMD_Operations::mul(Vmu, K_mu_array[index_ou][index_in][1][2]);
  SIMD_Type K_zx_mu =
      SIMD_Operations::mul(Vmu, K_mu_array[index_ou][index_in][2][0]);
  SIMD_Type K_zy_mu =
      SIMD_Operations::mul(Vmu, K_mu_array[index_ou][index_in][2][1]);
  SIMD_Type K_zz_mu =
      SIMD_Operations::mul(Vmu, K_mu_array[index_ou][index_in][2][2]);
  f[0] = SIMD_Operations::fmadd(K_xx_mu, Vu_x, f[0]);
  f[0] = SIMD_Operations::fmadd(K_xy_mu, Vu_y, f[0]);
  f[0] = SIMD_Operations::fmadd(K_xz_mu, Vu_z, f[0]);
  f[1] = SIMD_Operations::fmadd(K_yx_mu, Vu_x, f[1]);
  f[1] = SIMD_Operations::fmadd(K_yy_mu, Vu_y, f[1]);
  f[1] = SIMD_Operations::fmadd(K_yz_mu, Vu_z, f[1]);
  f[2] = SIMD_Operations::fmadd(K_zx_mu, Vu_x, f[2]);
  f[2] = SIMD_Operations::fmadd(K_zy_mu, Vu_y, f[2]);
  f[2] = SIMD_Operations::fmadd(K_zz_mu, Vu_z, f[2]);
  SIMD_Type K_xx_la =
      SIMD_Operations::mul(Vla, K_la_array[index_ou][index_in][0][0]);
  SIMD_Type K_xy_la =
      SIMD_Operations::mul(Vla, K_la_array[index_ou][index_in][0][1]);
  SIMD_Type K_xz_la =
      SIMD_Operations::mul(Vla, K_la_array[index_ou][index_in][0][2]);
  SIMD_Type K_yx_la =
      SIMD_Operations::mul(Vla, K_la_array[index_ou][index_in][1][0]);
  SIMD_Type K_yy_la =
      SIMD_Operations::mul(Vla, K_la_array[index_ou][index_in][1][1]);
  SIMD_Type K_yz_la =
      SIMD_Operations::mul(Vla, K_la_array[index_ou][index_in][1][2]);
  SIMD_Type K_zx_la =
      SIMD_Operations::mul(Vla, K_la_array[index_ou][index_in][2][0]);
  SIMD_Type K_zy_la =
      SIMD_Operations::mul(Vla, K_la_array[index_ou][index_in][2][1]);
  SIMD_Type K_zz_la =
      SIMD_Operations::mul(Vla, K_la_array[index_ou][index_in][2][2]);
  f[0] = SIMD_Operations::fmadd(K_xx_la, Vu_x, f[0]);
  f[0] = SIMD_Operations::fmadd(K_xy_la, Vu_y, f[0]);
  f[0] = SIMD_Operations::fmadd(K_xz_la, Vu_z, f[0]);
  f[1] = SIMD_Operations::fmadd(K_yx_la, Vu_x, f[1]);
  f[1] = SIMD_Operations::fmadd(K_yy_la, Vu_y, f[1]);
  f[1] = SIMD_Operations::fmadd(K_yz_la, Vu_z, f[1]);
  f[2] = SIMD_Operations::fmadd(K_zx_la, Vu_x, f[2]);
  f[2] = SIMD_Operations::fmadd(K_zy_la, Vu_y, f[2]);
  f[2] = SIMD_Operations::fmadd(K_zz_la, Vu_z, f[2]);
}
static void Multiply(Allocator_Type &allocator,
                     std::pair<const uint64_t *, unsigned> blocks,
                     T Struct_Type::*const u_fields[d],
                     T Struct_Type::*f_fields[d],
                     T Struct_Type::*const mu_field,
                     T Struct_Type::*const lambda_field,
                     T dx) {
  auto u_x = allocator.Get_Const_Array(u_fields[0]);
  auto u_y = allocator.Get_Const_Array(u_fields[1]);
  auto u_z = allocator.Get_Const_Array(u_fields[2]);
  auto f_x = allocator.Get_Array(f_fields[0]);
  auto f_y = allocator.Get_Array(f_fields[1]);
  auto f_z = allocator.Get_Array(f_fields[2]);
  auto mu = allocator.Get_Const_Array(mu_field);
  auto la = allocator.Get_Const_Array(lambda_field);
  SIMD_Type Vdx = SIMD_Operations::set(dx);
#pragma omp parallel for
  for (int b = 0; b < blocks.second; ++b) {
    auto offset = blocks.first[b];
    for (int e = 0; e < elements_per_block;
         e += SIMD_WIDTH, offset += SIMD_WIDTH * sizeof(T)) {
      SIMD_Type f[d];
      f[0] = SIMD_Operations::zero();
      f[1] = SIMD_Operations::zero();
      f[2] = SIMD_Operations::zero();
      Apply_Spoke<0, 0, 0, 0, 0, 0>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<0, 0, 0, 0, 0, 1>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<0, 0, 0, 0, 1, 0>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<0, 0, 0, 0, 1, 1>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<0, 0, 0, 1, 0, 0>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<0, 0, 0, 1, 0, 1>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<0, 0, 0, 1, 1, 0>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<0, 0, 0, 1, 1, 1>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<0, 0, 1, 0, 0, 0>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<0, 0, 1, 0, 0, 1>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<0, 0, 1, 0, 1, 0>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<0, 0, 1, 0, 1, 1>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<0, 0, 1, 1, 0, 0>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<0, 0, 1, 1, 0, 1>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<0, 0, 1, 1, 1, 0>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<0, 0, 1, 1, 1, 1>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<0, 1, 0, 0, 0, 0>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<0, 1, 0, 0, 0, 1>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<0, 1, 0, 0, 1, 0>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<0, 1, 0, 0, 1, 1>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<0, 1, 0, 1, 0, 0>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<0, 1, 0, 1, 0, 1>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<0, 1, 0, 1, 1, 0>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<0, 1, 0, 1, 1, 1>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<0, 1, 1, 0, 0, 0>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<0, 1, 1, 0, 0, 1>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<0, 1, 1, 0, 1, 0>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<0, 1, 1, 0, 1, 1>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<0, 1, 1, 1, 0, 0>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<0, 1, 1, 1, 0, 1>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<0, 1, 1, 1, 1, 0>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<0, 1, 1, 1, 1, 1>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<1, 0, 0, 0, 0, 0>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<1, 0, 0, 0, 0, 1>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<1, 0, 0, 0, 1, 0>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<1, 0, 0, 0, 1, 1>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<1, 0, 0, 1, 0, 0>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<1, 0, 0, 1, 0, 1>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<1, 0, 0, 1, 1, 0>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<1, 0, 0, 1, 1, 1>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<1, 0, 1, 0, 0, 0>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<1, 0, 1, 0, 0, 1>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<1, 0, 1, 0, 1, 0>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<1, 0, 1, 0, 1, 1>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<1, 0, 1, 1, 0, 0>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<1, 0, 1, 1, 0, 1>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<1, 0, 1, 1, 1, 0>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<1, 0, 1, 1, 1, 1>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<1, 1, 0, 0, 0, 0>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<1, 1, 0, 0, 0, 1>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<1, 1, 0, 0, 1, 0>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<1, 1, 0, 0, 1, 1>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<1, 1, 0, 1, 0, 0>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<1, 1, 0, 1, 0, 1>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<1, 1, 0, 1, 1, 0>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<1, 1, 0, 1, 1, 1>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<1, 1, 1, 0, 0, 0>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<1, 1, 1, 0, 0, 1>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<1, 1, 1, 0, 1, 0>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<1, 1, 1, 0, 1, 1>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<1, 1, 1, 1, 0, 0>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<1, 1, 1, 1, 0, 1>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<1, 1, 1, 1, 1, 0>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      Apply_Spoke<1, 1, 1, 1, 1, 1>(offset, f, u_x, u_y, u_z, mu, la, K_mu<T>,
                                    K_la<T>);
      f[0] = SIMD_Operations::mul(f[0], Vdx);
      f[1] = SIMD_Operations::mul(f[1], Vdx);
      f[2] = SIMD_Operations::mul(f[2], Vdx);
      SIMD_Operations::store(&f_x(offset), f[0]);
      SIMD_Operations::store(&f_y(offset), f[1]);
      SIMD_Operations::store(&f_z(offset), f[2]);
    }
  }
}