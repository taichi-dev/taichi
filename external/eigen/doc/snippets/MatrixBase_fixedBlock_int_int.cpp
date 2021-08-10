Matrix4d m = Vector4d(1,2,3,4).asDiagonal();
cout << "Here is the matrix m:" << endl << m << endl;
cout << "Here is m.fixed<2, 2>(2, 2):" << endl << m.block<2, 2>(2, 2) << endl;
m.block<2, 2>(2, 0) = m.block<2, 2>(2, 2);
cout << "Now the matrix m is:" << endl << m << endl;
