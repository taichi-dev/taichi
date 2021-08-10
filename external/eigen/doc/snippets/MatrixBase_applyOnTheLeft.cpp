Matrix3f A = Matrix3f::Random(3,3), B;
B << 0,1,0,  
     0,0,1,  
     1,0,0;
cout << "At start, A = " << endl << A << endl;
A.applyOnTheLeft(B); 
cout << "After applyOnTheLeft, A = " << endl << A << endl;
