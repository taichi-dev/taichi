#include "taichi/system/profiler.h"

TI_NAMESPACE_BEGIN

// A profiler's records form a tree structure
struct ProfilerRecordNode {
  std::vector<std::unique_ptr<ProfilerRecordNode>> childs;
  ProfilerRecordNode *parent;
  std::string name;
  float64 total_time;
  // Time per element
  bool account_tpe;
  uint64 total_elements;
  int64 num_samples;

  ProfilerRecordNode(const std::string &name, ProfilerRecordNode *parent) {
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
    TI_ASSERT(account_tpe);
    return total_time / (float64)total_elements;
  }

  ProfilerRecordNode *get_child(const std::string &name) {
    for (auto &ch : childs) {
      if (ch->name == name) {
        return ch.get();
      }
    }
    childs.push_back(std::make_unique<ProfilerRecordNode>(name, this));
    return childs.back().get();
  }
};

class ProfilerRecords {
 public:
  std::unique_ptr<ProfilerRecordNode> root;
  ProfilerRecordNode *current_node;
  int current_depth;
  bool enabled;

  ProfilerRecords(const std::string &name) {
    root = std::make_unique<ProfilerRecordNode>(
        fmt::format("[Profiler {}]", name), nullptr);
    current_node = root.get();
    current_depth = 0;  // depth(root) = 0
    enabled = true;
  }

  void print(ProfilerRecordNode *node, int depth);

  void print() {
    fmt::print(fg(fmt::color::cyan), std::string(80, '>') + "\n");
    print(root.get(), 0);
    fmt::print(fg(fmt::color::cyan), std::string(80, '>') + "\n");
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

  static ProfilerRecords &get_this_thread_instance() {
    // Use a raw pointer so that it lives together with the process
    static thread_local ProfilerRecords *profiler_records = nullptr;
    if (profiler_records == nullptr) {
      profiler_records = Profiling::get_instance().get_this_thread_profiler();
    }
    return *profiler_records;
  }
};

void ProfilerRecords::print(ProfilerRecordNode *node, int depth) {
  auto make_indent = [depth](int additional) {
    for (int i = 0; i < depth + additional; i++) {
      fmt::print("    ");
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
      return std::make_pair(1.0_f / 60_f, " m");
    } else {
      return std::make_pair(1.0_f / 3600_f, "h");
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
  fmt::color level_color;
  if (depth == 0)
    level_color = fmt::color::red;
  else if (depth == 1)
    level_color = fmt::color::green;
  else if (depth == 2)
    level_color = fmt::color::yellow;
  else if (depth == 3)
    level_color = fmt::color::blue;
  else if (depth >= 4)
    level_color = fmt::color::magenta;
  if (depth == 0) {
    // Root node only
    make_indent(0);
    fmt::print(fg(level_color), "{}\n", node->name.c_str());
  }
  if (total_time < 1e-6f) {
    for (auto &ch : node->childs) {
      make_indent(1);
      auto child_time = ch->total_time;
      auto bulk_statistics =
          fmt::format("{} {}", get_readable_time(child_time), ch->name);
      fmt::print(fg(level_color), "{:40}", bulk_statistics);
      fmt::print(fg(fmt::color::cyan), " [{} x {}]\n", ch->num_samples,
                 get_readable_time_with_scale(
                     ch->get_averaged(), get_time_scale(ch->get_averaged())));
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
      fmt::print(fg(level_color), "{:40}", bulk_statistics);
      fmt::print(fg(fmt::color::cyan), " [{} x {}]\n", ch->num_samples,
                 get_readable_time_with_scale(
                     ch->get_averaged(), get_time_scale(ch->get_averaged())));
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
      fmt::print(fg(level_color), "{} {:5.2f}%  {}\n",
                 get_readable_time_with_scale(unaccounted, scale),
                 unaccounted * 100.0 / total_time, "[unaccounted]");
    }
  }
}

ScopedProfiler::ScopedProfiler(std::string name, uint64 elements) {
  start_time = Time::get_time();
  this->name = name;
  this->elements = elements;
  stopped = false;
  ProfilerRecords::get_this_thread_instance().push(name);
}

void ScopedProfiler::stop() {
  TI_ASSERT_INFO(!stopped, "Profiler already stopped.");
  float64 elapsed = Time::get_time() - start_time;
  if ((int64)elements != -1) {
    ProfilerRecords::get_this_thread_instance().insert_sample(elapsed,
                                                              elements);
  } else {
    ProfilerRecords::get_this_thread_instance().insert_sample(elapsed);
  }
  ProfilerRecords::get_this_thread_instance().pop();
}

void ScopedProfiler::disable() {
  ProfilerRecords::get_this_thread_instance().enabled = false;
}

void ScopedProfiler::enable() {
  ProfilerRecords::get_this_thread_instance().enabled = true;
}

ScopedProfiler::~ScopedProfiler() {
  if (!stopped) {
    stop();
  }
}

Profiling &Profiling::get_instance() {
  static auto prof = new Profiling;
  return *prof;
}

ProfilerRecords *Profiling::get_this_thread_profiler() {
  std::lock_guard<std::mutex> _(mut);
  auto id = std::this_thread::get_id();
  std::stringstream ss;
  ss << id;
  if (profilers.find(id) == profilers.end()) {
    // Note: thread id may be reused
    profilers[id] = new ProfilerRecords(fmt::format("thread {}", ss.str()));
  }
  return profilers[id];
}

void Profiling::print_profile_info() {
  std::lock_guard<std::mutex> _(mut);
  for (auto p : profilers) {
    p.second->print();
  }
}

TI_NAMESPACE_END
