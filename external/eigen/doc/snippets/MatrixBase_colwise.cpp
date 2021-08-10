Matrix3d m = Matrix3d::Random();
cout << "Here is the matrix m:" << endl << m << endl;
cout << "Here is the sum of each column:" << endl << m.colwise().sum() << endl;
cout << "Here is the maximum absolute value of each column:"
     << endl << m.cwiseAbs().colwise().maxCoeff() << endl;
