/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

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

  void clear() {
    std::lock_guard<std::mutex> _(mut_);
    events_.clear();
  }

  void insert_event(const TimelineEvent &e) {
    std::lock_guard<std::mutex> _(mut_);
    events_.push_back(e);
  }

  std::vector<TimelineEvent> fetch_events() {
    std::lock_guard<std::mutex> _(mut_);
    std::vector<TimelineEvent> fetched;
    std::swap(fetched, events_);
    return fetched;
  }

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
  void insert_events(const std::vector<TimelineEvent> &events,
                     bool lock = true);

  static Timelines &get_instance();

  void clear();

  void save(const std::string &filename);

  void insert_timeline(Timeline *timeline) {
    std::lock_guard<std::mutex> _(mut_);
    timelines_.push_back(timeline);
  }

  void remove_timeline(Timeline *timeline) {
    std::lock_guard<std::mutex> _(mut_);
    std::remove(timelines_.begin(), timelines_.end(), timeline);
  }

 private:
  std::mutex mut_;
  std::vector<TimelineEvent> events_;
  std::vector<Timeline *> timelines_;
};

#define TI_TIMELINE(name) \
  taichi::Timeline::Guard _timeline_guard_##__LINE__(name);

#define TI_AUTO_TIMELINE TI_TIMELINE(__FUNCTION__)

TI_NAMESPACE_END
