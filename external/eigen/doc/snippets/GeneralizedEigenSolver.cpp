GeneralizedEigenSolver<MatrixXf> ges;
MatrixXf A = MatrixXf::Random(4,4);
MatrixXf B = MatrixXf::Random(4,4);
ges.compute(A, B);
cout << "The (complex) numerators of the generalzied eigenvalues are: " << ges.alphas().transpose() << endl;
cout << "The (real) denominatore of the generalzied eigenvalues are: " << ges.betas().transpose() << endl;
cout << "The (complex) generalzied eigenvalues are (alphas./beta): " << ges.eigenvalues().transpose() << endl;
