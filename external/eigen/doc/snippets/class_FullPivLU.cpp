typedef Matrix<double, 5, 3> Matrix5x3;
typedef Matrix<double, 5, 5> Matrix5x5;
Matrix5x3 m = Matrix5x3::Random();
cout << "Here is the matrix m:" << endl << m << endl;
Eigen::FullPivLU<Matrix5x3> lu(m);
cout << "Here is, up to permutations, its LU decomposition matrix:"
     << endl << lu.matrixLU() << endl;
cout << "Here is the L part:" << endl;
Matrix5x5 l = Matrix5x5::Identity();
l.block<5,3>(0,0).triangularView<StrictlyLower>() = lu.matrixLU();
cout << l << endl;
cout << "Here is the U part:" << endl;
Matrix5x3 u = lu.matrixLU().triangularView<Upper>();
cout << u << endl;
cout << "Let us now reconstruct the original matrix m:" << endl;
cout << lu.permutationP().inverse() * l * u * lu.permutationQ().inverse() << endl;
