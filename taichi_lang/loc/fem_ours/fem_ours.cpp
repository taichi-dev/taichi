Kernel(compute_Ap).def([&] {
  For(Ap(0), [&](Expr i, Expr j, Expr k) {
    auto cell_coord = Var(Vector({i, j, k}));
    auto Ku_tmp = Var(Vector(dim));
    Ku_tmp = Vector({0.0f, 0.0f, 0.0f});
    for (int cell = 0; cell < pow<dim>(2); cell++) {
      auto cell_offset =
          Var(Vector({-(cell / 4), -(cell / 2 % 2), -(cell % 2)}));
      auto cell_lambda = lambda[cell_coord + cell_offset];
      auto cell_mu = mu[cell_coord + cell_offset];
      for (int node = 0; node < pow<dim>(2); node++) {
        auto node_offset = Var(Vector({node / 4, node / 2 % 2, node % 2}));
        for (int u = 0; u < dim; u++)
          for (int v = 0; v < dim; v++)
            Ku_tmp(u) += (cell_lambda * K_la[cell][node][u][v] +
                          cell_mu * K_mu[cell][node][u][v]) *
                         p[cell_coord + cell_offset + node_offset](v);
      }
    }
  });
});
