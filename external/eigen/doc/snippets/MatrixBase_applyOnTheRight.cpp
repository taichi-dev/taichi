Matrix3f A = Matrix3f::Random(3,3), B;
B << 0,1,0,  
     0,0,1,  
     1,0,0;
cout << "At start, A = " << endl << A << endl;
A *= B;
cout << "After A *= B, A = " << endl << A << endl;
A.applyOnTheRight(B);  // equivalent to A *= B
cout << "After applyOnTheRight, A = " << endl << A << endl;
