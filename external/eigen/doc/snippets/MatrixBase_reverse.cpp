MatrixXi m = MatrixXi::Random(3,4);
cout << "Here is the matrix m:" << endl << m << endl;
cout << "Here is the reverse of m:" << endl << m.reverse() << endl;
cout << "Here is the coefficient (1,0) in the reverse of m:" << endl
     << m.reverse()(1,0) << endl;
cout << "Let us overwrite this coefficient with the value 4." << endl;
m.reverse()(1,0) = 4;
cout << "Now the matrix m is:" << endl << m << endl;
