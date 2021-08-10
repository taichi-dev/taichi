MatrixXd m(2,3);
m <<  2, -4, 6,
     -5,  1, 0;
cout << m.cwiseSign() << endl;
