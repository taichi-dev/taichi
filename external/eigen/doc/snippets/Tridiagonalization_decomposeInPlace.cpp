MatrixXd X = MatrixXd::Random(5,5);
MatrixXd A = X + X.transpose();
cout << "Here is a random symmetric 5x5 matrix:" << endl << A << endl << endl;

VectorXd diag(5);
VectorXd subdiag(4);
internal::tridiagonalization_inplace(A, diag, subdiag, true);
cout << "The orthogonal matrix Q is:" << endl << A << endl;
cout << "The diagonal of the tridiagonal matrix T is:" << endl << diag << endl;
cout << "The subdiagonal of the tridiagonal matrix T is:" << endl << subdiag << endl;
