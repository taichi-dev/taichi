MatrixXf matA(2,2); 
matA << 2, 0,  0, 2;
matA = matA * matA;
cout << matA;
