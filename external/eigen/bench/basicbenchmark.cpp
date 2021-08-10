
#include <iostream>
#include "BenchUtil.h"
#include "basicbenchmark.h"

int main(int argc, char *argv[])
{
  DISABLE_SSE_EXCEPTIONS();

  // this is the list of matrix type and size we want to bench:
  // ((suffix) (matrix size) (number of iterations))
  #define MODES ((3d)(3)(4000000)) ((4d)(4)(1000000)) ((Xd)(4)(1000000)) ((Xd)(20)(10000))
//   #define MODES ((Xd)(20)(10000))

  #define _GENERATE_HEADER(R,ARG,EL) << BOOST_PP_STRINGIZE(BOOST_PP_SEQ_HEAD(EL)) << "-" \
    << BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,EL)) << "x" \
    << BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,EL)) << "   /   "

  std::cout BOOST_PP_SEQ_FOR_EACH(_GENERATE_HEADER, ~, MODES ) << endl;

  const int tries = 10;

  #define _RUN_BENCH(R,ARG,EL) \
    std::cout << ARG( \
      BOOST_PP_CAT(Matrix, BOOST_PP_SEQ_HEAD(EL)) (\
         BOOST_PP_SEQ_ELEM(1,EL),BOOST_PP_SEQ_ELEM(1,EL)), BOOST_PP_SEQ_ELEM(2,EL), tries) \
    << "   ";

  BOOST_PP_SEQ_FOR_EACH(_RUN_BENCH, benchBasic<LazyEval>, MODES );
  std::cout << endl;
  BOOST_PP_SEQ_FOR_EACH(_RUN_BENCH, benchBasic<EarlyEval>, MODES );
  std::cout << endl;

  return 0;
}
