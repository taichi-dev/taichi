Matrix3d m = Matrix3d::Random();
cout << "Here is the matrix m:" << endl << m << endl;
cout << "Here is the norm of each column:" << endl << m.colwise().norm() << endl;
