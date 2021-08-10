#include <iostream>
#include <Eigen/Core>

using namespace Eigen;

#ifndef SCALAR
#define SCALAR float
#endif

#ifndef SIZE
#define SIZE 10000
#endif

#ifndef REPEAT
#define REPEAT 10000
#endif

typedef Matrix<SCALAR, Eigen::Dynamic, 1> Vec;

using namespace std;

SCALAR E_VDW(const Vec &interactions1, const Vec &interactions2)
{
  return (interactions2.cwise()/interactions1)
         .cwise().cube()
         .cwise().square()
         .cwise().square()
         .sum();
}

int main() 
{
  //
  //          1   2   3   4  ... (interactions)
  // ka       .   .   .   .  ...
  // rab      .   .   .   .  ...
  // energy   .   .   .   .  ...
  // ...     ... ... ... ... ...
  // (variables
  //    for
  // interaction)
  //
  Vec interactions1(SIZE), interactions2(SIZE); // SIZE is the number of vdw interactions in our system
  // SetupCalculations()
  SCALAR rab = 1.0;  
  interactions1.setConstant(2.4);
  interactions2.setConstant(rab);
  
  // Energy()
  SCALAR energy = 0.0;
  for (unsigned int i = 0; i<REPEAT; ++i) {
    energy += E_VDW(interactions1, interactions2);
    energy *= 1 + 1e-20 * i; // prevent compiler from optimizing the loop
  }
  cout << "energy = " << energy << endl;
}
