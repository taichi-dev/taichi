#ifndef __MATERIAL_H_
#define __MATERIAL_H_
#include <Setting.h>
#include <array>
//#include <variant>
namespace mn {
struct MaterialDynamics {
  MaterialDynamics() = delete;
  MaterialDynamics(int materialType);
  const int _materialType;
};
struct ElasticMaterialDynamics : public MaterialDynamics {
  ElasticMaterialDynamics() = delete;
  ElasticMaterialDynamics(int materialType, T ym, T pr, T d, T vol);
  T _youngsModulus, _poissonRatio, _density;
  T _lambda, _mu, _volume;
};
/// different from 'Model' class, 'Material' is composition of several aspects
struct Material {};
struct MaterialOptics {};
}  // namespace mn
#endif