import itertools

from microbenchmarks._items import AtomicOps, DataType
from microbenchmarks._metric import MetricType
from microbenchmarks._utils import get_ti_arch, tags2name

import taichi as ti


class Funcs():
    def __init__(self):
        self._funcs = {}

    def add_func(self, tag_list: list, func):
        self._funcs[tags2name(tag_list)] = {'tags': tag_list, 'func': func}

    def get_func(self, tags):
        for name, item in self._funcs.items():
            if set(item['tags']).issubset(tags):
                return item['func']
        return None


class BenchmarkPlan:
    def __init__(self, name='plan', arch='x64', basic_repeat_times=1):
        self.name = name
        self.arch = arch
        self.basic_repeat_times = basic_repeat_times
        self.info = {'name': self.name}
        self.plan = {}  # {'tags': [...], 'result': None}
        self.items = {}
        self.funcs = Funcs()

    def create_plan(self, *items):
        items_list = [[self.name]]
        for item in items:
            self.items[item.name] = item
            items_list.append(item.get_tags())
            self.info[item.name] = item.get_tags()
        case_list = list(itertools.product(*items_list))  #items generate cases
        for tags in case_list:
            self.plan[tags2name(tags)] = {'tags': tags, 'result': None}
        self._remove_conflict_items()

    def add_func(self, tag_list, func):
        self.funcs.add_func(tag_list, func)

    def run(self):
        for case, plan in self.plan.items():
            tag_list = plan['tags']
            MetricType.init_taichi(self.arch, tag_list)
            _ms = self.funcs.get_func(tag_list)(self.arch,
                                                self.basic_repeat_times,
                                                **self._get_kwargs(tag_list))
            plan['result'] = _ms
            print(f'{tag_list}={_ms}')
            ti.reset()
        rdict = {'results': self.plan, 'info': self.info}
        return rdict

    def _get_kwargs(self, tags, impl=True):
        kwargs = {}
        tags = tags[1:]  # tags = [case_name, item1_tag, item2_tag, ...]
        for item, tag in zip(self.items.values(), tags):
            kwargs[item.name] = item.impl(tag) if impl == True else tag
        return kwargs

    def _remove_conflict_items(self):
        remove_list = []
        #logical_atomic with float_type
        if set([AtomicOps.name, DataType.name]).issubset(self.items.keys()):
            for name, case in self.plan.items():
                kwargs_tag = self._get_kwargs(case['tags'], impl=False)
                atomic_tag = kwargs_tag[AtomicOps.name]
                dtype_tag = kwargs_tag[DataType.name]
                if not AtomicOps.is_supported_type(atomic_tag, dtype_tag):
                    remove_list.append(name)
        #remove
        for name in remove_list:
            self.plan.pop(name)

    def remove_cases_with_tags(self, tags: list):
        remove_list = []
        for case, plan in self.plan.items():
            if set(tags).issubset(plan['tags']):
                remove_list.append(case)
        #remove
        for case in remove_list:
            self.plan.pop(case)
