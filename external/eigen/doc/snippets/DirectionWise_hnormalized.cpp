typedef Matrix<double,4,Dynamic> Matrix4Xd;
Matrix4Xd M = Matrix4Xd::Random(4,5);
Projective3d P(Matrix4d::Random());
cout << "The matrix M is:" << endl << M << endl << endl;
cout << "M.colwise().hnormalized():" << endl << M.colwise().hnormalized() << endl << endl;
cout << "P*M:" << endl << P*M << endl << endl;
cout << "(P*M).colwise().hnormalized():" << endl << (P*M).colwise().hnormalized() << endl << endl;