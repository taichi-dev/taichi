SparseMatrix<double> A(3,3);
A.insert(1,2) = 0;
A.insert(0,1) = 1;
A.insert(2,0) = 2;
A.makeCompressed();
cout << "The matrix A is:" << endl << MatrixXd(A) << endl;
cout << "it has " << A.nonZeros() << " stored non zero coefficients that are: " << A.coeffs().transpose() << endl;
A.coeffs() += 10;
cout << "After adding 10 to every stored non zero coefficient, the matrix A is:" << endl << MatrixXd(A) << endl;
