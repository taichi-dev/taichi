Matrix3d m = Matrix3d::Zero();
m(0,2) = 1e-4;
cout << "Here's the matrix m:" << endl << m << endl;
cout << "m.isZero() returns: " << m.isZero() << endl;
cout << "m.isZero(1e-3) returns: " << m.isZero(1e-3) << endl;
