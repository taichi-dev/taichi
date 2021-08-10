// g++ -O3 -DNDEBUG -DMATSIZE=<x> benchmark.cpp -o benchmark && time ./benchmark

#include <iostream>

#include <Eigen/Core>

#ifndef MATSIZE
#define MATSIZE 3
#endif

using namespace std;
using namespace Eigen;

#ifndef REPEAT
#define REPEAT 40000000
#endif

#ifndef SCALAR
#define SCALAR double
#endif

int main(int argc, char *argv[])
{
    Matrix<SCALAR,MATSIZE,MATSIZE> I = Matrix<SCALAR,MATSIZE,MATSIZE>::Ones();
    Matrix<SCALAR,MATSIZE,MATSIZE> m;
    for(int i = 0; i < MATSIZE; i++)
        for(int j = 0; j < MATSIZE; j++)
        {
            m(i,j) = (i+MATSIZE*j);
        }
    asm("#begin");
    for(int a = 0; a < REPEAT; a++)
    {
        m = Matrix<SCALAR,MATSIZE,MATSIZE>::Ones() + 0.00005 * (m + (m*m));
    }
    asm("#end");
    cout << m << endl;
    return 0;
}
