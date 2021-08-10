int array[12];
for(int i = 0; i < 12; ++i) array[i] = i;
cout << Map<MatrixXi, 0, OuterStride<> >(array, 3, 3, OuterStride<>(4)) << endl;
