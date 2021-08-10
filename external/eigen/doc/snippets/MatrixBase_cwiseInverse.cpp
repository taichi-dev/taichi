MatrixXd m(2,3);
m << 2, 0.5, 1,   
     3, 0.25, 1;
cout << m.cwiseInverse() << endl;
