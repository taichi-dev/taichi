MatrixXd A(3,3);
A << 4,-1,2, -1,6,0, 2,0,5;
cout << "The matrix A is" << endl << A << endl;

LLT<MatrixXd> lltOfA(A); // compute the Cholesky decomposition of A
MatrixXd L = lltOfA.matrixL(); // retrieve factor L  in the decomposition
// The previous two lines can also be written as "L = A.llt().matrixL()"

cout << "The Cholesky factor L is" << endl << L << endl;
cout << "To check this, let us compute L * L.transpose()" << endl;
cout << L * L.transpose() << endl;
cout << "This should equal the matrix A" << endl;
