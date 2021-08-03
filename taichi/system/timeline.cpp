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

void Timeline::clear() {
  std::lock_guard<std::mutex> _(mut_);
  events_.clear();
}
void Timeline::insert_event(const TimelineEvent &e) {
  if (!Timelines::get_instance().get_enabled())
    return;
  std::lock_guard<std::mutex> _(mut_);
  events_.push_back(e);
}

std::vector<TimelineEvent> Timeline::fetch_events() {
  std::lock_guard<std::mutex> _(mut_);
  std::vector<TimelineEvent> fetched;
  std::swap(fetched, events_);
  return fetched;
}

Timeline::Guard::Guard(const std::string &name) : name_(name) {
  auto &timeline = Timeline::get_this_thread_instance();
  timeline.insert_event({name, true, Time::get_time(), timeline.tid_});
}

Timeline::Guard::~Guard() {
  auto &timeline = Timeline::get_this_thread_instance();
  timeline.insert_event({name_, false, Time::get_time(), timeline.tid_});
}

void Timelines::insert_events(const std::vector<TimelineEvent> &events) {
  std::lock_guard<std::mutex> _(mut_);
  insert_events_without_locking(events);
}

void Timelines::insert_events_without_locking(
    const std::vector<TimelineEvent> &events) {
  events_.insert(events_.end(), events.begin(), events.end());
}

Timelines &taichi::Timelines::get_instance() {
  static auto instance = new Timelines();
  return *instance;
}

void Timelines::clear() {
  std::lock_guard<std::mutex> _(mut_);
  events_.clear();
  for (auto timeline : timelines_) {
    timeline->clear();
  }
}

void Timelines::save(const std::string &filename) {
  std::lock_guard<std::mutex> _(mut_);
  std::sort(timelines_.begin(), timelines_.end(), [](Timeline *a, Timeline *b) {
    return a->get_name() < b->get_name();
  });
  for (auto timeline : timelines_) {
    insert_events_without_locking(timeline->fetch_events());
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

void Timelines::insert_timeline(Timeline *timeline) {
  std::lock_guard<std::mutex> _(mut_);
  timelines_.push_back(timeline);
}

void Timelines::remove_timeline(Timeline *timeline) {
  std::lock_guard<std::mutex> _(mut_);
  trash(std::remove(timelines_.begin(), timelines_.end(), timeline));
}

bool Timelines::get_enabled() {
  return enabled_;
}

void Timelines::set_enabled(bool enabled) {
  enabled_ = enabled;
}

TI_NAMESPACE_END
