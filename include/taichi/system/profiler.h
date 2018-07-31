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

  void print(Node *node, int depth) {
    auto make_indent = [depth](int additional) {
      for (int i = 0; i < depth + additional; i++) {
        fmt::print("  ");
      }
    };
    using TimeScale = std::pair<real, std::string>;

    auto get_time_scale = [&](real t) -> TimeScale {
      if (t < 1e-6) {
        return std::make_pair(1e9_f, "ns");
      } else if (t < 1e-3) {
        return std::make_pair(1e6_f, "us");
      } else if (t < 1) {
        return std::make_pair(1e3_f, "ms");
      } else if (t < 60) {
        return std::make_pair(1_f, " s");
      } else if (t < 3600) {
        return std::make_pair(60_f, " m");
      } else {
        return std::make_pair(3600_f, "h");
      }
    };

    auto get_readable_time_with_scale = [&](real t, TimeScale scale) {
      return fmt::format("{:7.3f} {}", t * scale.first, scale.second);
    };

    auto get_readable_time = [&](real t) {
      auto scale = get_time_scale(t);
      return get_readable_time_with_scale(t, scale);
    };

    float64 total_time = node->total_time;
    if (depth == 0) {
      // Root node only
      make_indent(0);
      fmt::print_colored(fmt::GREEN, "{}\n", node->name.c_str());
    }
    if (total_time < 1e-6f) {
      for (auto &ch : node->childs) {
        make_indent(1);
        auto child_time = ch->total_time;
        auto bulk_statistics =
            fmt::format("{} {}", get_readable_time(child_time), ch->name);
        fmt::print_colored(fmt::YELLOW, "{:40}", bulk_statistics);
        fmt::print_colored(
            fmt::CYAN, " [{} x {}]\n", ch->num_samples,
            get_readable_time_with_scale(ch->get_averaged(),
                                         get_time_scale(ch->get_averaged())));
        print(ch.get(), depth + 1);
      }
    } else {
      TimeScale scale = get_time_scale(total_time);
      float64 unaccounted = total_time;
      for (auto &ch : node->childs) {
        make_indent(1);
        auto child_time = ch->total_time;
        std::string bulk_statistics = fmt::format(
            "{} {:5.2f}%  {}", get_readable_time_with_scale(child_time, scale),
            child_time * 100.0 / total_time, ch->name);
        fmt::print_colored(fmt::YELLOW, "{:40}", bulk_statistics);
        fmt::print_colored(
            fmt::CYAN, " [{} x {}]\n", ch->num_samples,
            get_readable_time_with_scale(ch->get_averaged(),
                                         get_time_scale(ch->get_averaged())));
        if (ch->account_tpe) {
          make_indent(1);
          fmt::print("                     [TPE] {}\n",
                     get_readable_time(ch->total_time));
        }
        print(ch.get(), depth + 1);
        unaccounted -= child_time;
      }
      if (!node->childs.empty() && (unaccounted > total_time * 0.005)) {
        make_indent(1);
        fmt::print_colored(fmt::BLUE, "{} {:5.2f}%  {}\n",
                           get_readable_time_with_scale(unaccounted, scale),
                           unaccounted * 100.0 / total_time, "[unaccounted]");
      }
    }
  }

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

inline void print_profile_info() {
  ProfilerRecords::get_instance().print();
}

TC_NAMESPACE_END
