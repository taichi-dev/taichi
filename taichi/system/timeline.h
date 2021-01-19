#pragma once

#include <vector>
#include <mutex>

#include "taichi/common/core.h"
#include "taichi/system/timer.h"

TI_NAMESPACE_BEGIN

struct TimelineEvent {
  std::string name;
  bool begin;
  float64 time;
  std::string tid;

  std::string to_json();
};

class Timeline {
 public:
  Timeline();

  ~Timeline();

  static Timeline &get_this_thread_instance();

  void set_name(const std::string &tid) {
    tid_ = tid;
  }

  std::string get_name() {
    return tid_;
  }

  void clear();

  void insert_event(const TimelineEvent &e);

  std::vector<TimelineEvent> fetch_events();

  class Guard {
   public:
    Guard(const std::string &name);

    ~Guard();

   private:
    std::string name_;
  };

 private:
  std::string tid_;
  std::mutex mut_;
  std::vector<TimelineEvent> events_;
};

// A timeline system for multi-threaded applications
class Timelines {
 public:
  static Timelines &get_instance();

  void insert_events(const std::vector<TimelineEvent> &events);

  void insert_events_without_locking(const std::vector<TimelineEvent> &events);

  void insert_timeline(Timeline *timeline);

  void remove_timeline(Timeline *timeline);

  void clear();

  void save(const std::string &filename);

  bool get_enabled();

  void set_enabled(bool enabled);

 private:
  std::mutex mut_;
  std::vector<TimelineEvent> events_;
  std::vector<Timeline *> timelines_;
  bool enabled_{false};
};

#define TI_TIMELINE(name) \
  taichi::Timeline::Guard _timeline_guard_##__LINE__(name);

#define TI_AUTO_TIMELINE TI_TIMELINE(__FUNCTION__)

TI_NAMESPACE_END
