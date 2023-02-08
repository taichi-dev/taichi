#pragma once

#include "taichi/system/timeline.h"
#include "taichi/program/kernel_profiler.h"
#include "taichi/rhi/amdgpu/amdgpu_driver.h"
#include "taichi/rhi/amdgpu/amdgpu_context.h"

#include <string>
#include <stdint.h>

namespace taichi::lang {
    class EventToolkitAMDGPU;

    class KernelProfilerAMDGPU : public KernelProfilerBase {
        public:
            std::string get_device_name() override;

            bool reinit_with_metrics(const std::vector<std::string> metrics) override;
            void trace(KernelProfilerBase::TaskHandle &task_handle,
                        const std::string &kernel_name,
                        void *kernel,
                        uint32_t grid_size,
                        uint32_t block_size,
                        uint32_t dynamic_smem_size);
            void sync() override;
            void update() override;
            void clear() override;
            void stop(KernelProfilerBase::TaskHandle handle) override;

            bool set_profiler_toolkit(std::string toolkit_name) override;

            bool statistics_on_traced_records();

            KernelProfilerBase::TaskHandle start_with_handle(
                const std::string &kernel_name) override;

        private:
            std::unique_ptr<EventToolkitAMDGPU> event_toolkit_{nullptr};
    };

    class EventToolkitAMDGPU : public EventToolkitBase {
        public:
            void update_record(uint32_t records_size_after_sync,
                        std::vector<KernelProfileTracedRecord> &traced_records) override;
            KernelProfilerBase::TaskHandle start_with_handle(
                const std::string &kernel_name) override;
            void update_timeline(std::vector<KernelProfileTracedRecord> &traced_records) override;
    };
}
