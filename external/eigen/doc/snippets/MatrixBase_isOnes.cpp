Matrix3d m = Matrix3d::Ones();
m(0,2) += 1e-4;
cout << "Here's the matrix m:" << endl << m << endl;
cout << "m.isOnes() returns: " << m.isOnes() << endl;
cout << "m.isOnes(1e-3) returns: " << m.isOnes(1e-3) << endl;
