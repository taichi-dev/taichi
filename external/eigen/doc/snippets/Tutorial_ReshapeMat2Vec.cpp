MatrixXf M1(3,3);    // Column-major storage
M1 << 1, 2, 3,
      4, 5, 6,
      7, 8, 9;

Map<RowVectorXf> v1(M1.data(), M1.size());
cout << "v1:" << endl << v1 << endl;

Matrix<float,Dynamic,Dynamic,RowMajor> M2(M1);
Map<RowVectorXf> v2(M2.data(), M2.size());
cout << "v2:" << endl << v2 << endl;