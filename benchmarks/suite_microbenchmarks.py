import os
import time

from microbenchmarks import benchmark_plan_list
from utils import dump2json


class MicroBenchmark:
    suite_name = "microbenchmarks"
    config = {
        "cuda": {"enable": False},
        "vulkan": {"enable": False},
        "opengl": {"enable": False},
        "amdgpu": {"enable": True},
    }

    def __init__(self):
        self._results = {}
        self._info = {}

    def get_benchmark_info(self):
        info_dict = {}
        arch_list = []
        for arch, item in self.config.items():
            if item["enable"] == True:
                arch_list.append(arch)
        info_dict["archs"] = arch_list
        return info_dict
    
    def str_to_plan(self, plan_str):
        match plan_str:
            case "AtomicOpsPlan":
                return benchmark_plan_list[0]
            case "FillPlan":
                return benchmark_plan_list[1]
            case "MathOpsPlan":
                return benchmark_plan_list[2]
            case "MatrixOpsPlan":
                return benchmark_plan_list[3]
            case "MemcpyPlan":
                return benchmark_plan_list[4]
            case "SaxpyPlan":
                return benchmark_plan_list[5]
            case "Stencil2DPlan":
                return benchmark_plan_list[6]

    def run(self, arch, plan_str):
        arch_results = {}
        self._info[arch] = {}
        plan = self.str_to_plan(plan_str)
        plan_impl = plan(arch)
        results = plan_impl.run()
        self._info[arch][plan_impl.name] = results["info"]
        arch_results[plan_impl.name] = results["results"]

        self._results[arch] = arch_results

    def save_as_json(self, suite_dir="./"):
        for arch in self._results:
            arch_dir = os.path.join(suite_dir, arch)
            os.makedirs(arch_dir, exist_ok=True)
            self._save_info_as_json(arch, arch_dir)
            self._save_cases_as_json(arch, arch_dir)

    def _save_info_as_json(self, arch, arch_dir="./"):
        info_path = os.path.join(arch_dir, "_info.json")
        with open(info_path, "w") as f:
            print(dump2json(self._info[arch]), file=f)

    def _save_cases_as_json(self, arch, arch_dir="./"):
        for case in self._info[arch]:
            case_path = os.path.join(arch_dir, (case + ".json"))
            case_results = self._results[arch][case]
            with open(case_path, "w") as f:
                case_str = dump2json(case_results)
                print(case_str, file=f)
