Matrix2i m = Matrix2i::Random();
cout << "Here is the matrix m:" << endl << m << endl;
cout << "Here is the transpose of m:" << endl << m.transpose() << endl;
cout << "Here is the coefficient (1,0) in the transpose of m:" << endl
     << m.transpose()(1,0) << endl;
cout << "Let us overwrite this coefficient with the value 0." << endl;
m.transpose()(1,0) = 0;
cout << "Now the matrix m is:" << endl << m << endl;
