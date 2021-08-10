Matrix3i m = Matrix3i::Random();
cout << "Here is the matrix m:" << endl << m << endl;
cout << "Here is the symmetric matrix extracted from the upper part of m:" << endl
     << Matrix3i(m.selfadjointView<Upper>()) << endl;
cout << "Here is the symmetric matrix extracted from the lower part of m:" << endl
     << Matrix3i(m.selfadjointView<Lower>()) << endl;
