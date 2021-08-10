Array3d v(1,2,3);
v(1) *= 0.0/0.0;
v(2) /= 0.0;
cout << v << endl << endl;
cout << isfinite(v) << endl;
