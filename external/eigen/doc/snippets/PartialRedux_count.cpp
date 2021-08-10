Matrix3d m = Matrix3d::Random();
cout << "Here is the matrix m:" << endl << m << endl;
Matrix<ptrdiff_t, 3, 1> res = (m.array() >= 0.5).rowwise().count();
cout << "Here is the count of elements larger or equal than 0.5 of each row:" << endl;
cout << res << endl;
