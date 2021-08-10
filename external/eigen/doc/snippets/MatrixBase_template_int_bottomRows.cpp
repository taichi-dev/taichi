Array44i a = Array44i::Random();
cout << "Here is the array a:" << endl << a << endl;
cout << "Here is a.bottomRows<2>():" << endl;
cout << a.bottomRows<2>() << endl;
a.bottomRows<2>().setZero();
cout << "Now the array a is:" << endl << a << endl;
