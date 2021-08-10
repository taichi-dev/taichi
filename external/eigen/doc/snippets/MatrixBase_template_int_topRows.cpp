Array44i a = Array44i::Random();
cout << "Here is the array a:" << endl << a << endl;
cout << "Here is a.topRows<2>():" << endl;
cout << a.topRows<2>() << endl;
a.topRows<2>().setZero();
cout << "Now the array a is:" << endl << a << endl;
