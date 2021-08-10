// g++ -O3 -DNDEBUG benchmarkX.cpp -o benchmarkX && time ./benchmarkX

#include <iostream>
#include <Eigen/Core>

using namespace std;
using namespace Eigen;

#ifndef VECTYPE
#define VECTYPE VectorXLd
#endif

#ifndef VECSIZE
#define VECSIZE 1000000
#endif

#ifndef REPEAT
#define REPEAT 1000
#endif

int main(int argc, char *argv[])
{
	VECTYPE I = VECTYPE::Ones(VECSIZE);
	VECTYPE m(VECSIZE,1);
	for(int i = 0; i < VECSIZE; i++)
	{
		m[i] = 0.1 * i/VECSIZE;
	}
	for(int a = 0; a < REPEAT; a++)
	{
		m = VECTYPE::Ones(VECSIZE) + 0.00005 * (m.cwise().square() + m/4);
	}
	cout << m[0] << endl;
	return 0;
}
