#include "taichi/ir/type_system.h"

namespace taichi {
namespace lang {

void TyVar::unify(DataType dt, std::map<Identifier, DataType> &solutions) {
  if (solutions.find(name) != solutions.end()) {
    if (solutions[name] != dt) {
      throw TyVarMismatch(name, solutions[name], dt);
    }
  } else {
    solutions[name] = dt;
  }
}

DataType TyVar::resolve(const std::map<Identifier, DataType> &solutions) {
  if (solutions.find(name) == solutions.end()) {
    throw TyVarUnsolved(name);
  } else {
    return solutions.at(name);
  }
}

std::string TyVar::to_string() {
  return name.name();
}

void TyLub::unify(DataType dt, std::map<Identifier, DataType> &solutions) {
  TyMono(resolve(solutions)).unify(dt, solutions);
}

DataType TyLub::resolve(const std::map<Identifier, DataType> &solutions) {
  return promoted_type(lhs->resolve(solutions)->get_compute_type(), rhs->resolve(solutions)->get_compute_type());
}

std::string TyLub::to_string() {
  return lhs->to_string() + " | " + rhs->to_string();
}

void TyCompute::unify(DataType dt, std::map<Identifier, DataType> &solutions) {
  TyMono(resolve(solutions)).unify(dt, solutions);
}

DataType TyCompute::resolve(const std::map<Identifier, DataType> &solutions) {
  return exp->resolve(solutions)->get_compute_type();
}

std::string TyCompute::to_string() {
  return "comp(" + exp->to_string() + ")";
}

void TyMono::unify(DataType dt, std::map<Identifier, DataType> &solutions) {
  if (monotype != dt) {
    throw TypeMismatch(monotype, dt);
  }
}

DataType TyMono::resolve(const std::map<Identifier, DataType> &solutions) {
  return monotype;
}

std::string TyMono::to_string() {
  return monotype.to_string();
}

std::string TypeMismatch::to_string() {
  return "the argument type " + arg.to_string() + " is incompatible with the expected type " + param.to_string();
}

std::string TyVarUnsolved::to_string() {
  return "cannot infer the type variable " + var.name() + ". this is not supposed to happen; please report this as a bug";
}

std::string TraitMismatch::to_string() {
  return "the argument type " + dt.to_string() + " is not " + trait->to_string();
}

std::string ArgLengthMismatch::to_string() {
  return std::to_string(arg) + " arguments were passed in but expected " + std::to_string(param) + ". this is not supposed to happen; please report this as a bug";
}

DataType Signature::type_check(std::vector<DataType> arguments) {
  if (parameters.size() != arguments.size()) {
    throw ArgLengthMismatch(parameters.size(), arguments.size());
  }
  std::map<Identifier, DataType> solutions;
  for (int i = 0; i < parameters.size(); i++) {
    parameters[i]->unify(arguments[i], solutions);
  }
  for (auto &c: constraints) {
    auto dt = c.tyvar->resolve(solutions);
    if (!c.trait->validate(dt)) {
        throw TraitMismatch(dt, c.trait);
    }
  }
  return ret_type->resolve(solutions);
}

}
}
