Matrix3d m = Matrix3d::Random();
cout << "Here is the matrix m:" << endl << m << endl;
cout << "Here is the square norm of each row:" << endl << m.rowwise().squaredNorm() << endl;
