MatrixXd m(3,4);
m.resize(5, NoChange);
cout << "m: " << m.rows() << " rows, " << m.cols() << " cols" << endl;
