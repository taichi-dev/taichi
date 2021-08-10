#include <Eigen/Array>

int main(int argc, char *argv[])
{
  std::cout.precision(2);

  // demo static functions
  Eigen::Matrix3f m3 = Eigen::Matrix3f::Random();
  Eigen::Matrix4f m4 = Eigen::Matrix4f::Identity();

  std::cout << "*** Step 1 ***\nm3:\n" << m3 << "\nm4:\n" << m4 << std::endl;

  // demo non-static set... functions
  m4.setZero();
  m3.diagonal().setOnes();
  
  std::cout << "*** Step 2 ***\nm3:\n" << m3 << "\nm4:\n" << m4 << std::endl;

  // demo fixed-size block() expression as lvalue and as rvalue
  m4.block<3,3>(0,1) = m3;
  m3.row(2) = m4.block<1,3>(2,0);

  std::cout << "*** Step 3 ***\nm3:\n" << m3 << "\nm4:\n" << m4 << std::endl;

  // demo dynamic-size block()
  {
    int rows = 3, cols = 3;
    m4.block(0,1,3,3).setIdentity();
    std::cout << "*** Step 4 ***\nm4:\n" << m4 << std::endl;
  }

  // demo vector blocks
  m4.diagonal().block(1,2).setOnes();
  std::cout << "*** Step 5 ***\nm4.diagonal():\n" << m4.diagonal() << std::endl;
  std::cout << "m4.diagonal().start(3)\n" << m4.diagonal().start(3) << std::endl;

  // demo coeff-wise operations
  m4 = m4.cwise()*m4;
  m3 = m3.cwise().cos();
  std::cout << "*** Step 6 ***\nm3:\n" << m3 << "\nm4:\n" << m4 << std::endl;

  // sums of coefficients
  std::cout << "*** Step 7 ***\n m4.sum(): " << m4.sum() << std::endl;
  std::cout << "m4.col(2).sum(): " << m4.col(2).sum() << std::endl;
  std::cout << "m4.colwise().sum():\n" << m4.colwise().sum() << std::endl;
  std::cout << "m4.rowwise().sum():\n" << m4.rowwise().sum() << std::endl;

  // demo intelligent auto-evaluation
  m4 = m4 * m4; // auto-evaluates so no aliasing problem (performance penalty is low)
  Eigen::Matrix4f other = (m4 * m4).lazy(); // forces lazy evaluation
  m4 = m4 + m4; // here Eigen goes for lazy evaluation, as with most expressions
  m4 = -m4 + m4 + 5 * m4; // same here, Eigen chooses lazy evaluation for all that.
  m4 = m4 * (m4 + m4); // here Eigen chooses to first evaluate m4 + m4 into a temporary.
                       // indeed, here it is an optimization to cache this intermediate result.
  m3 = m3 * m4.block<3,3>(1,1); // here Eigen chooses NOT to evaluate block() into a temporary
    // because accessing coefficients of that block expression is not more costly than accessing
    // coefficients of a plain matrix.
  m4 = m4 * m4.transpose(); // same here, lazy evaluation of the transpose.
  m4 = m4 * m4.transpose().eval(); // forces immediate evaluation of the transpose

  std::cout << "*** Step 8 ***\nm3:\n" << m3 << "\nm4:\n" << m4 << std::endl;
}
