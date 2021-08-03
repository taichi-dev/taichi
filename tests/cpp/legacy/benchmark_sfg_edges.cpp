#include <algorithm>
#include <chrono>  // std::chrono::system_clock
#include <random>  // std::default_random_engine
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "taichi/system/profiler.h"
#include "taichi/util/testing.h"

TLANG_NAMESPACE_BEGIN

using PairData = std::pair<int, int>;

struct Empty {};
using UnorderedMapSet = std::unordered_map<int, std::unordered_set<int>>;
using LLVMVecSet = llvm::SmallVector<std::pair<int, llvm::SmallSet<int, 4>>, 4>;
using LLVMVecVec =
    llvm::SmallVector<std::pair<int, llvm::SmallVector<int, 4>>, 4>;
using StlVecSet = std::vector<std::pair<int, std::unordered_set<int>>>;
using StlVecVec = std::vector<std::pair<int, std::vector<int>>>;

using FlattenSet = llvm::SmallSet<PairData, 16>;
using FlattenVec = llvm::SmallVector<PairData, 16>;

// https://stackoverflow.com/a/1055563/12003165
// Over-engineering a bit. typeid().name() is mangled and hard to read...
template <typename T>
struct TypeNameTraits;

#define REGISTER_TYPE_NAME(X) \
  template <>                 \
  struct TypeNameTraits<X> {  \
    static const char *name;  \
  };                          \
  const char *TypeNameTraits<X>::name = #X

REGISTER_TYPE_NAME(LLVMVecSet);
REGISTER_TYPE_NAME(LLVMVecVec);
REGISTER_TYPE_NAME(StlVecSet);
REGISTER_TYPE_NAME(StlVecVec);

void insert(const PairData &d, Empty *m) {
  TI_PROFILER("Empty insert");
}

bool lookup(const PairData &d, const Empty &m, const std::string &found) {
  TI_PROFILER("Empty lookup " + found);
  return false;
}

void insert(const PairData &d, UnorderedMapSet *m) {
  TI_PROFILER("UnorderedMapSet insert");
  (*m)[d.first].insert(d.second);
}

bool lookup(const PairData &d,
            const UnorderedMapSet &m,
            const std::string &found) {
  TI_PROFILER("UnorderedMapSet lookup " + found);
  auto itr = m.find(d.first);
  if (itr == m.end()) {
    return false;
  }
  return itr->second.count(d.second) > 0;
}

template <typename VecSet>
void insert_vecset(const PairData &d, VecSet *m) {
  auto pname = fmt::format("{} insert", TypeNameTraits<VecSet>::name);
  TI_PROFILER(pname);
  int i1 = 0;
  for (; i1 < m->size(); ++i1) {
    if ((*m)[i1].first == d.first) {
      break;
    }
  }
  if (i1 == m->size()) {
    m->push_back({});
    m->back().first = d.first;
  }
  (*m)[i1].second.insert(d.second);
}

template <typename VecSet>
bool lookup_vecset(const PairData &d,
                   const VecSet &m,
                   const std::string &found) {
  auto pname = fmt::format("{} lookup {}", TypeNameTraits<VecSet>::name, found);
  TI_PROFILER(pname);
  int i1 = 0;
  for (; i1 < m.size(); ++i1) {
    if (m[i1].first == d.first) {
      break;
    }
  }
  if (i1 == m.size()) {
    return false;
  }
  return m[i1].second.count(d.second) > 0;
}

template <typename VecVec>
void insert_vecvec(const PairData &d, VecVec *m) {
  auto pname = fmt::format("{} insert", TypeNameTraits<VecVec>::name);
  TI_PROFILER(pname);
  int i1 = 0;
  for (; i1 < m->size(); ++i1) {
    if ((*m)[i1].first == d.first) {
      break;
    }
  }
  if (i1 == m->size()) {
    m->push_back({});
    m->back().first = d.first;
  }

  auto &v2 = (*m)[i1].second;
  for (int i2 = 0; i2 < v2.size(); ++i2) {
    if (v2[i2] == d.second) {
      return;
    }
  }
  v2.push_back(d.second);
}

template <typename VecVec>
bool lookup_vecvec(const PairData &d,
                   const VecVec &m,
                   const std::string &found) {
  auto pname = fmt::format("{} lookup {}", TypeNameTraits<VecVec>::name, found);
  TI_PROFILER(pname);
  int i1 = 0;
  for (; i1 < m.size(); ++i1) {
    if (m[i1].first == d.first) {
      break;
    }
  }
  if (i1 == m.size()) {
    return false;
  }
  const auto &v2 = m[i1].second;
  for (int i2 = 0; i2 < v2.size(); ++i2) {
    if (v2[i2] == d.second) {
      return true;
    }
  }
  return false;
}

template <typename C>
void insert(const PairData &d, C *m) {
  if constexpr (std::is_same_v<C, LLVMVecSet> || std::is_same_v<C, StlVecSet>) {
    insert_vecset(d, m);
  } else if constexpr (std::is_same_v<C, LLVMVecVec> ||
                       std::is_same_v<C, StlVecVec>) {
    insert_vecvec(d, m);
  }
}

template <typename C>
bool lookup(const PairData &d, const C &m, const std::string &found) {
  if constexpr (std::is_same_v<C, LLVMVecSet> || std::is_same_v<C, StlVecSet>) {
    return lookup_vecset(d, m, found);
  } else if constexpr (std::is_same_v<C, LLVMVecVec> ||
                       std::is_same_v<C, StlVecVec>) {
    return lookup_vecvec(d, m, found);
  }
}

void insert(const PairData &d, FlattenSet *m) {
  TI_PROFILER("FlattenSet insert");
  m->insert(d);
}

bool lookup(const PairData &d, const FlattenSet &m, const std::string &found) {
  TI_PROFILER("FlattenSet lookup " + found);
  return m.count(d) > 0;
}

bool lookup(const PairData &d, const FlattenVec &m, const std::string &found) {
  TI_PROFILER("FlattenVec lookup " + found);

  auto itr = std::lower_bound(m.begin(), m.end(), d);
  return (itr != m.end()) && (*itr == d);
}

void insert(const PairData &d, FlattenVec *m) {
  TI_PROFILER("FlattenVec insert");

  auto itr = std::lower_bound(m->begin(), m->end(), d);
  if ((itr != m->end()) && (*itr == d)) {
    return;
  }

  m->insert(itr, d);
  // m->push_back(d);
  // std::sort(m->begin(), m->end());
}

template <typename M>
void run_test(const std::vector<PairData> &data,
              const std::vector<PairData> &non_exists) {
  for (int i = 0; i < 10; ++i) {
    M m;
    for (const auto &p : data) {
      insert(p, &m);
    }

    for (const auto &p : data) {
      bool l = lookup(p, m, "found");
      if constexpr (!std::is_same_v<M, Empty>) {
        TI_CHECK(l);
      }
    }

    for (const auto &p : non_exists) {
      bool l = lookup(p, m, "not found");
      TI_CHECK(!l);
    }
  }
}

// Basic tests within a basic block
TI_TEST("benchmark_sfg") {
#if 0
  std::vector<PairData> data = {
      {0, 0}, {0, 2}, {0, 2}, {0, 5},                  // 0
      {1, 1}, {1, 2}, {1, 3}, {1, 3}, {1, 6}, {1, 6},  // 1
      {2, 0}, {2, 0}, {2, 2},                          // 2
      {3, 5}, {3, 6}, {3, 7},                          // 3
      {5, 0}, {5, 2}, {5, 2}, {5, 2}, {5, 2},          // 5
      {6, 1}, {6, 2}, {6, 2},                          // 6
      {9, 9},                                          // 9
  };
  std::vector<PairData> non_exists = {
      {0, 1}, {0, 3},          // 0
      {1, 4}, {1, 5},          // 1
      {3, 0}, {3, 1}, {3, 2},  // 3
      {4, 1},                  // 4
      {6, 3},                  // 6
      {7, 8},                  // 7
  };
  const auto seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine rng(seed);
  std::shuffle(data.begin(), data.end(), rng);
  std::shuffle(non_exists.begin(), non_exists.end(), rng);

  SECTION("Empty") {
    // Get an idea of the overhead of the profiler itself.
    run_test<Empty>(data, non_exists);
  }

  SECTION("UnorderedMapSet") {
    run_test<UnorderedMapSet>(data, non_exists);
  }

  SECTION("LLVMVecSet") {
    run_test<LLVMVecSet>(data, non_exists);
  }

  SECTION("LLVMVecVec") {
    run_test<LLVMVecVec>(data, non_exists);
  }

  SECTION("StlVecSet") {
    run_test<StlVecSet>(data, non_exists);
  }

  SECTION("StlVecVec") {
    run_test<StlVecVec>(data, non_exists);
  }

  SECTION("FlattenSet") {
    run_test<FlattenSet>(data, non_exists);
  }

  SECTION("FlattenVec") {
    run_test<FlattenVec>(data, non_exists);
  }
  Profiling::get_instance().print_profile_info();
#endif
}

TLANG_NAMESPACE_END
