MatrixXf M1 = MatrixXf::Random(3,8);
cout << "Column major input:" << endl << M1 << "\n";
Map<MatrixXf,0,OuterStride<> > M2(M1.data(), M1.rows(), (M1.cols()+2)/3, OuterStride<>(M1.outerStride()*3));
cout << "1 column over 3:" << endl << M2 << "\n";

typedef Matrix<float,Dynamic,Dynamic,RowMajor> RowMajorMatrixXf;
RowMajorMatrixXf M3(M1);
cout << "Row major input:" << endl << M3 << "\n";
Map<RowMajorMatrixXf,0,Stride<Dynamic,3> > M4(M3.data(), M3.rows(), (M3.cols()+2)/3,
                                              Stride<Dynamic,3>(M3.outerStride(),3));
cout << "1 column over 3:" << endl << M4 << "\n";