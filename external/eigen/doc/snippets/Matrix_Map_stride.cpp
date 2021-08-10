Matrix4i A;
A << 1,  2,  3,  4,
     5,  6,  7,  8,
     9, 10, 11, 12,
    13, 14, 15, 16;

std::cout << Matrix2i::Map(&A(1,1),Stride<8,2>()) << std::endl;
