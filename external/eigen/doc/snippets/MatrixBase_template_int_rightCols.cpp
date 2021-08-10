Array44i a = Array44i::Random();
cout << "Here is the array a:" << endl << a << endl;
cout << "Here is a.rightCols<2>():" << endl;
cout << a.rightCols<2>() << endl;
a.rightCols<2>().setZero();
cout << "Now the array a is:" << endl << a << endl;
