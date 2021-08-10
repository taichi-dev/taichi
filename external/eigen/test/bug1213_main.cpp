
// This is a regression unit regarding a weird linking issue with gcc.

#include "bug1213.h"

int main()
{
  return 0;
}


template<typename T, int dim>
bool bug1213_2(const Eigen::Matrix<T,dim,1>& )
{
  return true;
}

template bool bug1213_2<float,3>(const Eigen::Vector3f&);
