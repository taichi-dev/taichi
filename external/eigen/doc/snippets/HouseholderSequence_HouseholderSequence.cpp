Matrix3d v = Matrix3d::Random();
cout << "The matrix v is:" << endl;
cout << v << endl;

Vector3d v0(1, v(1,0), v(2,0));
cout << "The first Householder vector is: v_0 = " << v0.transpose() << endl;
Vector3d v1(0, 1, v(2,1));
cout << "The second Householder vector is: v_1 = " << v1.transpose()  << endl;
Vector3d v2(0, 0, 1);
cout << "The third Householder vector is: v_2 = " << v2.transpose() << endl;

Vector3d h = Vector3d::Random();
cout << "The Householder coefficients are: h = " << h.transpose() << endl;

Matrix3d H0 = Matrix3d::Identity() - h(0) * v0 * v0.adjoint();
cout << "The first Householder reflection is represented by H_0 = " << endl;
cout << H0 << endl;
Matrix3d H1 = Matrix3d::Identity() - h(1) * v1 * v1.adjoint();
cout << "The second Householder reflection is represented by H_1 = " << endl;
cout << H1 << endl;
Matrix3d H2 = Matrix3d::Identity() - h(2) * v2 * v2.adjoint();
cout << "The third Householder reflection is represented by H_2 = " << endl;
cout << H2 << endl;
cout << "Their product is H_0 H_1 H_2 = " << endl;
cout << H0 * H1 * H2 << endl;

HouseholderSequence<Matrix3d, Vector3d> hhSeq(v, h);
Matrix3d hhSeqAsMatrix(hhSeq);
cout << "If we construct a HouseholderSequence from v and h" << endl;
cout << "and convert it to a matrix, we get:" << endl;
cout << hhSeqAsMatrix << endl;
