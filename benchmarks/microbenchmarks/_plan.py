import itertools

from microbenchmarks._results import Results
from microbenchmarks._utils import get_ti_arch, tags2name

import taichi as ti


class BenchmarkPlan:
    def __init__(self, name='plan', arch='x64', basic_repeat_times=1):
        self.name = name
        self.arch = arch
        self.basic_repeat_times = basic_repeat_times
        self.info = {'name': self.name}
        self.plan = {}  # {'tags': [...], 'result': None}
        self.items = []
        self.func = None

    def create_plan(self, *items):
        items_list = [[self.name]]
        for item in items:
            self.items.append(item())
            items_list.append(item().get_tags())
            self.info[item.name] = item().get_tags()
        case_list = list(itertools.product(*items_list))  #items generate case
        for tags in case_list:
            self.plan[tags2name(tags)] = {'tags': tags, 'result': None}

    def _get_func_agrs(self, tags):
        impl_list = []
        for item in self.items:
            for tag in tags:
                if item.tag_in_item(tag):
                    impl_list.append(item.impl(tag))
        return tuple(impl_list)

    def _init_taichi(self, arch, tags):
        for tag in tags:
            if Results.init_taichi(arch, tag):
                return True
        return False

    def set_func(self, func):
        self.func = func

    def _run_func(self, tags: list):
        return self.func(self.arch, self.basic_repeat_times,
                         *self._get_func_agrs(tags))

    def run(self):
        for case, plan in self.plan.items():
            tag_list = plan['tags']
            self._init_taichi(self.arch, tag_list)
            _ms = self._run_func(tag_list)
            plan['result'] = _ms
            print(f'{tag_list}={_ms}')
            ti.reset()
        rdict = {'results': self.plan, 'info': self.info}
        return rdict

    def print_plan(self):
        for name in self.plan:
            print(name)
