MatrixXi m(2,2);
m << 1, 0,
     1, 1;
cout << "Comparing m with identity matrix:" << endl;
cout << m.cwiseNotEqual(MatrixXi::Identity(2,2)) << endl;
Index count = m.cwiseNotEqual(MatrixXi::Identity(2,2)).count();
cout << "Number of coefficients that are not equal: " << count << endl;
