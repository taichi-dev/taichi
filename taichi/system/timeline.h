/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

#include "taichi/common/core.h"
#include "taichi/system/timer.h"
#include "spdlog/fmt/bundled/color.h"

TI_NAMESPACE_BEGIN

// TODO: cppize
struct TimelineEvent {
  std::string name;
  bool begin;
  float64 time;
  std::string tid;

  std::string to_json() {
    std::string json{"{"};
    json += fmt::format("\"cat\":\"taichi\",");
    json += fmt::format("\"pid\":0,");
    json += fmt::format("\"tid\":\"{}\",", tid);
    json += fmt::format("\"ph\":\"{}\",", begin ? "B" : "E");
    json += fmt::format("\"name\":\"{}\",", name);
    json += fmt::format("\"ts\":\"{}\"", uint64(time * 1000000));
    json += "}";
    return json;
  }
};

class Timeline {
 public:
  Timeline() : tid_("unnamed") {
  }

  void set_name(const std::string &tid) {
    tid_ = tid;
  }

  static Timeline &get_this_thread_instance() {
    thread_local Timeline instance;
    return instance;
  }

  ~Timeline();

  void insert_event(const TimelineEvent &e) {
    events_.push_back(e);
  }

  class Guard {
   public:
    Guard(const std::string &name) : name_(name) {
      auto &timeline = Timeline::get_this_thread_instance();
      timeline.insert_event({name, true, Time::get_time(), timeline.tid_});
    }

    ~Guard() {
      auto &timeline = Timeline::get_this_thread_instance();
      timeline.insert_event({name_, false, Time::get_time(), timeline.tid_});
    }

   private:
    std::string name_;
  };

 private:
  std::string tid_;
  std::vector<TimelineEvent> events_;
};

// A timeline system for multi-threaded applications
class Timelines {
 public:
  void clear() {
    std::lock_guard<std::mutex> _(mut_);
    events_.clear();
  }

  void insert_events(const std::vector<TimelineEvent> &events) {
    std::lock_guard<std::mutex> _(mut_);
    events_.insert(events_.begin(), events.begin(), events.end());
  }

  static Timelines &get_instance() {
    static Timelines instance;
    return instance;
  }

  ~Timelines() {
    std::ofstream fout("timeline.json");
    fout << "[";
    bool first = true;
    for (auto &e : events_) {
      if (first) {
        first = false;
      } else {
        fout << ",";
      }
      fout << e.to_json() << std::endl;
    }
    fout << "]";
  }

 private:
  std::mutex mut_;
  std::vector<TimelineEvent> events_;
};

inline Timeline::~Timeline() {
  Timelines::get_instance().insert_events(events_);
}

#define TI_TIMELINE(name) \
  taichi::Timeline::Guard _timeline_guard_##__LINE__(name);

#define TI_AUTO_TIMELINE TI_TIMELINE(__FUNCTION__)

TI_NAMESPACE_END
