MatrixXi m = MatrixXi::Random(2,3);
cout << "Here is the matrix m:" << endl << m << endl;
cout << "m.replicate<3,2>() = ..." << endl;
cout << m.replicate<3,2>() << endl;
