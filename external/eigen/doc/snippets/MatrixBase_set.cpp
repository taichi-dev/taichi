Matrix3i m1;
m1 << 1, 2, 3,
      4, 5, 6,
      7, 8, 9;
cout << m1 << endl << endl;
Matrix3i m2 = Matrix3i::Identity();
m2.block(0,0, 2,2) << 10, 11, 12, 13;
cout << m2 << endl << endl;
Vector2i v1;
v1 << 14, 15;
m2 << v1.transpose(), 16,
      v1, m1.block(1,1,2,2);
cout << m2 << endl;
