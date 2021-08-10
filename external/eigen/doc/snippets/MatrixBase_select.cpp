MatrixXi m(3, 3);
m << 1, 2, 3,
     4, 5, 6,
     7, 8, 9;
m = (m.array() >= 5).select(-m, m);
cout << m << endl;
