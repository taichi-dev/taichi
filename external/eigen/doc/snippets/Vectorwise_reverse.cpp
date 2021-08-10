MatrixXi m = MatrixXi::Random(3,4);
cout << "Here is the matrix m:" << endl << m << endl;
cout << "Here is the rowwise reverse of m:" << endl << m.rowwise().reverse() << endl;
cout << "Here is the colwise reverse of m:" << endl << m.colwise().reverse() << endl;

cout << "Here is the coefficient (1,0) in the rowise reverse of m:" << endl
<< m.rowwise().reverse()(1,0) << endl;
cout << "Let us overwrite this coefficient with the value 4." << endl;
//m.colwise().reverse()(1,0) = 4;
cout << "Now the matrix m is:" << endl << m << endl;
