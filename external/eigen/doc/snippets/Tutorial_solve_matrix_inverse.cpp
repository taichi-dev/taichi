Matrix3f A;
Vector3f b;
A << 1,2,3,  4,5,6,  7,8,10;
b << 3, 3, 4;
Vector3f x = A.inverse() * b;
cout << "The solution is:" << endl << x << endl;
