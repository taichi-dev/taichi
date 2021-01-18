#include "taichi/system/timeline.h"

TI_NAMESPACE_BEGIN

std::string TimelineEvent::to_json() {
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

Timeline::Timeline() : tid_("unnamed") {
  Timelines::get_instance().insert_timeline(this);
}

Timeline &Timeline::get_this_thread_instance() {
  thread_local Timeline instance;
  return instance;
}

Timeline::~Timeline() {
  Timelines::get_instance().insert_events(fetch_events());
  Timelines::get_instance().remove_timeline(this);
}

Timeline::Guard::Guard(const std::string &name) : name_(name) {
  auto &timeline = Timeline::get_this_thread_instance();
  timeline.insert_event({name, true, Time::get_time(), timeline.tid_});
}

Timeline::Guard::~Guard() {
  auto &timeline = Timeline::get_this_thread_instance();
  timeline.insert_event({name_, false, Time::get_time(), timeline.tid_});
}

void Timelines::insert_events(const std::vector<TimelineEvent> &events,
                              bool lock) {
  if (lock) {
    std::lock_guard<std::mutex> _(mut_);
    events_.insert(events_.begin(), events.begin(), events.end());
  } else {
    events_.insert(events_.begin(), events.begin(), events.end());
  }
}

Timelines &taichi::Timelines::get_instance() {
  static Timelines instance;
  return instance;
}

void Timelines::clear() {
  // TODO: also clear events of each time line
  std::lock_guard<std::mutex> _(mut_);
  events_.clear();
  for (auto timeline : timelines_) {
    timeline->clear();
  }
}

void Timelines::save(const std::string &filename) {
  std::lock_guard<std::mutex> _(mut_);
  for (auto timeline : timelines_) {
    insert_events(timeline->fetch_events(), false);
  }
  if (!ends_with(filename, ".json")) {
    TI_WARN("Timeline filename {} should end with '.json'.", filename);
  }
  std::ofstream fout(filename);
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

TI_NAMESPACE_END
