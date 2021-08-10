#include "main.h"

#include <exception>  // std::exception

struct Foo
{
  static Index object_count;
  static Index object_limit;
  int dummy;

  Foo() : dummy(0)
  {
#ifdef EIGEN_EXCEPTIONS
    // TODO: Is this the correct way to handle this?
    if (Foo::object_count > Foo::object_limit) { std::cout << "\nThrow!\n"; throw Foo::Fail(); }
#endif
	  std::cout << '+';
    ++Foo::object_count;
  }

  ~Foo()
  {
	  std::cout << '-';
    --Foo::object_count;
  }

  class Fail : public std::exception {};
};

Index Foo::object_count = 0;
Index Foo::object_limit = 0;

#undef EIGEN_TEST_MAX_SIZE
#define EIGEN_TEST_MAX_SIZE 3

void test_ctorleak()
{
  typedef Matrix<Foo, Dynamic, Dynamic> MatrixX;
  typedef Matrix<Foo, Dynamic, 1> VectorX;
  
  Foo::object_count = 0;
  for(int i = 0; i < g_repeat; i++) {
    Index rows = internal::random<Index>(2,EIGEN_TEST_MAX_SIZE), cols = internal::random<Index>(2,EIGEN_TEST_MAX_SIZE);
    Foo::object_limit = rows*cols;
    {
    MatrixX r(rows, cols);
    Foo::object_limit = r.size()+internal::random<Index>(0, rows*cols - 2);
    std::cout << "object_limit =" << Foo::object_limit << std::endl;
#ifdef EIGEN_EXCEPTIONS
    try
    {
#endif
      if(internal::random<bool>()) {
        std::cout <<       "\nMatrixX m(" << rows << ", " << cols << ");\n";
        MatrixX m(rows, cols);
      }
      else {
        std::cout <<       "\nMatrixX m(r);\n";
        MatrixX m(r);
      }
#ifdef EIGEN_EXCEPTIONS
      VERIFY(false);  // not reached if exceptions are enabled
    }
    catch (const Foo::Fail&) { /* ignore */ }
#endif
    }
    VERIFY_IS_EQUAL(Index(0), Foo::object_count);

    {
      Foo::object_limit = (rows+1)*(cols+1);
      MatrixX A(rows, cols);
      VERIFY_IS_EQUAL(Foo::object_count, rows*cols);
      VectorX v=A.row(0);
      VERIFY_IS_EQUAL(Foo::object_count, (rows+1)*cols);
      v = A.col(0);
      VERIFY_IS_EQUAL(Foo::object_count, rows*(cols+1));
    }
    VERIFY_IS_EQUAL(Index(0), Foo::object_count);
  }
  std::cout << "\n";
}
