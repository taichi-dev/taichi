// g++ -fopenmp -I .. -O3 -DNDEBUG -finline-limit=1000 benchmarkX.cpp -o b && time ./b

#include <iostream>

#include <Eigen/Core>

using namespace std;
using namespace Eigen;

#ifndef MATTYPE
#define MATTYPE MatrixXLd
#endif

#ifndef MATSIZE
#define MATSIZE 400
#endif

#ifndef REPEAT
#define REPEAT 100
#endif

int main(int argc, char *argv[])
{
	MATTYPE I = MATTYPE::Ones(MATSIZE,MATSIZE);
	MATTYPE m(MATSIZE,MATSIZE);
	for(int i = 0; i < MATSIZE; i++) for(int j = 0; j < MATSIZE; j++)
	{
		m(i,j) = (i+j+1)/(MATSIZE*MATSIZE);
	}
	for(int a = 0; a < REPEAT; a++)
	{
		m = I + 0.0001 * (m + m*m);
	}
	cout << m(0,0) << endl;
	return 0;
}
