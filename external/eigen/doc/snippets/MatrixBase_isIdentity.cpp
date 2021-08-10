Matrix3d m = Matrix3d::Identity();
m(0,2) = 1e-4;
cout << "Here's the matrix m:" << endl << m << endl;
cout << "m.isIdentity() returns: " << m.isIdentity() << endl;
cout << "m.isIdentity(1e-3) returns: " << m.isIdentity(1e-3) << endl;
