#include <taichi/system/profiler.h>

TC_NAMESPACE_BEGIN

void ProfilerRecords::print(ProfilerRecords::Node *node, int depth) {
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
  fmt::Color level_color;
  if (depth == 0)
    level_color = fmt::RED;
  else if (depth == 1)
    level_color = fmt::GREEN;
  else if (depth == 2)
    level_color = fmt::YELLOW;
  else if (depth == 3)
    level_color = fmt::BLUE;
  else if (depth >= 4)
    level_color = fmt::MAGENTA;
  if (depth == 0) {
    // Root node only
    make_indent(0);
    fmt::print_colored(level_color, "{}\n", node->name.c_str());
  }
  if (total_time < 1e-6f) {
    for (auto &ch : node->childs) {
      make_indent(1);
      auto child_time = ch->total_time;
      auto bulk_statistics =
          fmt::format("{} {}", get_readable_time(child_time), ch->name);
      fmt::print_colored(level_color, "{:40}", bulk_statistics);
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
      fmt::print_colored(level_color, "{:40}", bulk_statistics);
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
      fmt::print_colored(level_color, "{} {:5.2f}%  {}\n",
                         get_readable_time_with_scale(unaccounted, scale),
                         unaccounted * 100.0 / total_time, "[unaccounted]");
    }
  }
}

TC_NAMESPACE_END
