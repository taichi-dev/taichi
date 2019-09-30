#include "Material.h"
#include <MnBase/Meta/AllocMeta.h>
namespace mn {
MaterialDynamics::MaterialDynamics(int materialType)
    : _materialType(materialType) {
}
/// Elastic (neohookean & fixed corotated)
ElasticMaterialDynamics::ElasticMaterialDynamics(int materialType,
                                                 T ym,
                                                 T pr,
                                                 T d,
                                                 T vol)
    : MaterialDynamics(materialType),
      _youngsModulus(ym),
      _poissonRatio(pr),
      _density(d),
      _volume(vol) {
  _lambda = _youngsModulus * _poissonRatio /
            ((1 + _poissonRatio) * (1 - 2 * _poissonRatio));
  _mu = _youngsModulus / (2 * (1 + _poissonRatio));
}
}  // namespace mn