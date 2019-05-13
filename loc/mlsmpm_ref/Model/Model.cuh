#ifndef __MODEL_CUH_
#define __MODEL_CUH_
#include <Setting.h>
#include <memory>
#include <Simulation/Material/Material.h>
#include <Simulation/Geometry/Geometry.h>
namespace mn {
class Model {
 public:
  auto &refGeometryPtr() {
    return _geometry;
  }
  auto &refMaterialDynamicsPtr() {
    return _materialDynamics;
  }
  Model() = delete;
  Model(std::unique_ptr<Model> model)
      : _geometry(std::move(model->refGeometryPtr())),
        _materialDynamics(std::move(model->refMaterialDynamicsPtr())) {
  }
  Model(std::unique_ptr<Geometry> geometry,
        std::unique_ptr<MaterialDynamics> materialD)
      : _geometry(std::move(geometry)),
        _materialDynamics(std::move(materialD)) {
  }
 private:
  std::unique_ptr<Geometry> _geometry;
  std::unique_ptr<MaterialDynamics> _materialDynamics;
};
}  // namespace mn
#endif