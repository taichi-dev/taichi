MatrixXf a(2,3); a << 1, 2, 3, 4, 5, 6;
cout << "Here is the initial matrix a:\n" << a << endl;


a.transposeInPlace();
cout << "and after being transposed:\n" << a << endl;