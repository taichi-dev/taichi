/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/common/util.h>
#include <taichi/system/timer.h>
#include <vector>
#include <map>
#include <memory>
#if defined(TC_PLATFORM_WINDOWS)
#undef max
#endif

TC_NAMESPACE_BEGIN

class ProfilerRecords {
 public:
  struct Node {
    std::vector<std::unique_ptr<Node>> childs;
    Node *parent;
    std::string name;
    float64 total_time;
    // Time per element
    bool account_tpe;
    uint64 total_elements;
    int64 num_samples;

    Node(const std::string &name, Node *parent) {
      this->name = name;
      this->parent = parent;
      this->total_time = 0.0_f64;
      this->num_samples = 0ll;
      this->total_elements = 0ll;
      this->account_tpe = false;
    }

    void insert_sample(float64 sample) {
      num_samples += 1;
      total_time += sample;
    }

    void insert_sample(float64 sample, uint64 elements) {
      account_tpe = true;
      num_samples += 1;
      total_time += sample;
      total_elements += elements;
    }

    float64 get_averaged() const {
      return total_time / (float64)std::max(num_samples, int64(1));
    }

    float64 get_averaged_tpe() const {
      TC_ASSERT(account_tpe);
      return total_time / (float64)total_elements;
    }

    Node *get_child(const std::string &name) {
      for (auto &ch : childs) {
        if (ch->name == name) {
          return ch.get();
        }
      }
      childs.push_back(std::make_unique<Node>(name, this));
      return childs.back().get();
    }
  };

  std::unique_ptr<Node> root;
  Node *current_node;
  int current_depth;
  bool enabled;

  ProfilerRecords() {
    root = std::make_unique<Node>("[Profiler]", nullptr);
    current_node = root.get();
    current_depth = 0;  // depth(root) = 0
    enabled = true;
  }

  void print(Node *node, int depth);

  void print() {
    fmt::print_colored(fmt::CYAN, std::string(80, '>') + "\n");
    print(root.get(), 0);
    fmt::print_colored(fmt::CYAN, std::string(80, '>') + "\n");
  }

  void insert_sample(float64 time) {
    if (!enabled)
      return;
    current_node->insert_sample(time);
  }

  void insert_sample(float64 time, uint64 tpe) {
    if (!enabled)
      return;
    current_node->insert_sample(time, tpe);
  }

  void push(const std::string name) {
    if (!enabled)
      return;
    current_node = current_node->get_child(name);
    current_depth += 1;
  }

  void pop() {
    if (!enabled)
      return;
    current_node = current_node->parent;
    current_depth -= 1;
  }

  static ProfilerRecords &get_instance() {
    static ProfilerRecords profiler_records;
    return profiler_records;
  }
};

class Profiler {
 public:
  float64 start_time;
  std::string name;
  bool stopped;
  uint64 elements;

  Profiler(std::string name, uint64 elements = -1) {
    start_time = Time::get_time();
    this->name = name;
    this->elements = elements;
    stopped = false;
    ProfilerRecords::get_instance().push(name);
  }

  void stop() {
    assert_info(!stopped, "Profiler already stopped.");
    float64 elapsed = Time::get_time() - start_time;
    if ((int64)elements != -1) {
      ProfilerRecords::get_instance().insert_sample(elapsed, elements);
    } else {
      ProfilerRecords::get_instance().insert_sample(elapsed);
    }
    ProfilerRecords::get_instance().pop();
  }

  ~Profiler() {
    if (!stopped) {
      stop();
    }
  }

  static void disable() {
    ProfilerRecords::get_instance().enabled = false;
  }

  static void enable() {
    ProfilerRecords::get_instance().enabled = true;
  }
};

#define TC_PROFILE(name, statements) \
  {                                  \
    taichi::Profiler _(name);        \
    statements;                      \
  }

#define TC_PROFILER(name) taichi::Profiler _profiler_##__LINE__(name);

#define TC_PROFILE_TPE(name, statements, elements) \
  {                                                \
    taichi::Profiler _(name, elements);            \
    statements;                                    \
  }

#define TI_AUTO_PROF TC_PROFILER(__FUNCTION__)

inline void print_profile_info() {
  ProfilerRecords::get_instance().print();
}

TC_NAMESPACE_END
