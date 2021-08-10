Matrix2f m = Matrix2f::Random();
m = (m + m.adjoint()).eval();
JacobiRotation<float> J;
J.makeJacobi(m, 0, 1);
cout << "Here is the matrix m:" << endl << m << endl;
m.applyOnTheLeft(0, 1, J.adjoint());
m.applyOnTheRight(0, 1, J);
cout << "Here is the matrix J' * m * J:" << endl << m << endl;