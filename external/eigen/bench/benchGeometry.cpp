#include <iostream>
#include <iomanip>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <bench/BenchTimer.h>

using namespace Eigen;
using namespace std;

#ifndef REPEAT
#define REPEAT 1000000
#endif

enum func_opt
{
    TV,
    TMATV,
    TMATVMAT,
};


template <class res, class arg1, class arg2, int opt>
struct func;

template <class res, class arg1, class arg2>
struct func<res, arg1, arg2, TV>
{
    static EIGEN_DONT_INLINE res run( arg1& a1, arg2& a2 )
    {
	asm ("");
	return a1 * a2;
    }
};

template <class res, class arg1, class arg2>
struct func<res, arg1, arg2, TMATV>
{
    static EIGEN_DONT_INLINE res run( arg1& a1, arg2& a2 )
    {
	asm ("");
	return a1.matrix() * a2;
    }
};

template <class res, class arg1, class arg2>
struct func<res, arg1, arg2, TMATVMAT>
{
    static EIGEN_DONT_INLINE res run( arg1& a1, arg2& a2 )
    {
	asm ("");
	return res(a1.matrix() * a2.matrix());
    }
};

template <class func, class arg1, class arg2>
struct test_transform
{
    static void run()
    {
	arg1 a1;
	a1.setIdentity();
	arg2 a2;
	a2.setIdentity();

	BenchTimer timer;
	timer.reset();
	for (int k=0; k<10; ++k)
	{
	    timer.start();
	    for (int k=0; k<REPEAT; ++k)
		a2 = func::run( a1, a2 );
	    timer.stop();
	}
	cout << setprecision(4) << fixed << timer.value() << "s  " << endl;;
    }
};


#define run_vec( op, scalar, mode, option, vsize ) \
    std::cout << #scalar << "\t " << #mode << "\t " << #option << " " << #vsize " "; \
    {\
	typedef Transform<scalar, 3, mode, option> Trans;\
	typedef Matrix<scalar, vsize, 1, option> Vec;\
	typedef func<Vec,Trans,Vec,op> Func;\
	test_transform< Func, Trans, Vec >::run();\
    }

#define run_trans( op, scalar, mode, option ) \
    std::cout << #scalar << "\t " << #mode << "\t " << #option << "   "; \
    {\
	typedef Transform<scalar, 3, mode, option> Trans;\
	typedef func<Trans,Trans,Trans,op> Func;\
	test_transform< Func, Trans, Trans >::run();\
    }

int main(int argc, char* argv[])
{
    cout << "vec = trans * vec" << endl;
    run_vec(TV, float,  Isometry, AutoAlign, 3);
    run_vec(TV, float,  Isometry, DontAlign, 3);
    run_vec(TV, float,  Isometry, AutoAlign, 4);
    run_vec(TV, float,  Isometry, DontAlign, 4);
    run_vec(TV, float,  Projective, AutoAlign, 4);
    run_vec(TV, float,  Projective, DontAlign, 4);
    run_vec(TV, double, Isometry, AutoAlign, 3);
    run_vec(TV, double, Isometry, DontAlign, 3);
    run_vec(TV, double, Isometry, AutoAlign, 4);
    run_vec(TV, double, Isometry, DontAlign, 4);
    run_vec(TV, double, Projective, AutoAlign, 4);
    run_vec(TV, double, Projective, DontAlign, 4);

    cout << "vec = trans.matrix() * vec" << endl;
    run_vec(TMATV, float,  Isometry, AutoAlign, 4);
    run_vec(TMATV, float,  Isometry, DontAlign, 4);
    run_vec(TMATV, double, Isometry, AutoAlign, 4);
    run_vec(TMATV, double, Isometry, DontAlign, 4);

    cout << "trans = trans1 * trans" << endl;
    run_trans(TV, float,  Isometry, AutoAlign);
    run_trans(TV, float,  Isometry, DontAlign);
    run_trans(TV, double, Isometry, AutoAlign);
    run_trans(TV, double, Isometry, DontAlign);
    run_trans(TV, float,  Projective, AutoAlign);
    run_trans(TV, float,  Projective, DontAlign);
    run_trans(TV, double, Projective, AutoAlign);
    run_trans(TV, double, Projective, DontAlign);

    cout << "trans = trans1.matrix() * trans.matrix()" << endl;
    run_trans(TMATVMAT, float,  Isometry, AutoAlign);
    run_trans(TMATVMAT, float,  Isometry, DontAlign);
    run_trans(TMATVMAT, double, Isometry, AutoAlign);
    run_trans(TMATVMAT, double, Isometry, DontAlign);
}

