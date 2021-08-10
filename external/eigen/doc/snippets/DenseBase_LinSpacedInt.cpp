cout << "Even spacing inputs:" << endl;
cout << VectorXi::LinSpaced(8,1,4).transpose() << endl;
cout << VectorXi::LinSpaced(8,1,8).transpose() << endl;
cout << VectorXi::LinSpaced(8,1,15).transpose() << endl;
cout << "Uneven spacing inputs:" << endl;
cout << VectorXi::LinSpaced(8,1,7).transpose() << endl;
cout << VectorXi::LinSpaced(8,1,9).transpose() << endl;
cout << VectorXi::LinSpaced(8,1,16).transpose() << endl;
