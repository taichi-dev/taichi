#include "common.h"

class SampleApp : public App {
 public:
  SampleApp() : App(1920, 1080, "Sample 1: Window") {
  }

  std::vector<StreamSemaphore> render_loop(
      StreamSemaphore image_available_semaphore) override {
    return {};
  }

 public:
};

int main() {
  std::unique_ptr<SampleApp> app = std::make_unique<SampleApp>();
  app->run();

  return 0;
}
