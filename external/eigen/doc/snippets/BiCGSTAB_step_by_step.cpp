  int n = 10000;
  VectorXd x(n), b(n);
  SparseMatrix<double> A(n,n);
  /* ... fill A and b ... */ 
  BiCGSTAB<SparseMatrix<double> > solver(A);
  // start from a random solution
  x = VectorXd::Random(n);
  solver.setMaxIterations(1);
  int i = 0;
  do {
    x = solver.solveWithGuess(b,x);
    std::cout << i << " : " << solver.error() << std::endl;
    ++i;
  } while (solver.info()!=Success && i<100);