Matrix3f A;
Vector3f b;
A << 1,2,3,  0,5,6,  0,0,10;
b << 3, 3, 4;
A.triangularView<Upper>().solveInPlace(b);
cout << "The solution is:" << endl << b << endl;
