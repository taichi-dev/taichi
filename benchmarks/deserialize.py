import json
import jsbeautifier
import os

from copy import deepcopy
# from utils import dump2json
def dump2json(obj):
    obj2dict = obj if type(obj) is dict else obj.__dict__
    options = jsbeautifier.default_options()
    options.indent_size = 4
    return jsbeautifier.beautify(json.dumps(obj2dict), options)

class ResultsBuilder():
    def __init__(self, results_file_path=None):
        self._suites_result = {}
        if results_file_path is not None:
            self._file_path = results_file_path
        else:
            print('Enter path of result file:')
            self._file_path = input()
        self.load_suites_result()

    def load_suites_result(self):
        # benchmark info
        info_path = os.path.join(self._file_path, '_info.json')
        with open(info_path, 'r') as f:
            info_dict = json.load(f)['suites']
            # suite info
            for suite_name, attrs in info_dict.items():
                self._suites_result[suite_name] = {}
                for arch in attrs['archs']:
                    self._suites_result[suite_name][arch] = {}
                    suite_info_path = os.path.join(self._file_path, suite_name, arch, "_info.json")
                    with open(suite_info_path, 'r') as f:
                        suite_info_dict = json.load(f)
                    # case info
                    for case_name in suite_info_dict:
                        items = suite_info_dict[case_name]
                        items.pop('name')
                        items['metrics'] = items.pop('get_metric')   
                        self._suites_result[suite_name][arch][case_name] = {'items': items}
        # cases result
        for suite_name in self._suites_result:
            for arch in self._suites_result[suite_name]:
                for case_name in self._suites_result[suite_name][arch]:
                    case_info_path = os.path.join(self._file_path, suite_name, arch, case_name + ".json")
                    with open(case_info_path, 'r') as f:
                        case_results = json.load(f)
                        remove_none_list = []
                        for name,data in case_results.items():
                            # data['metric'] = data['tags'][-1]
                            data['tags'] = data['tags'][1:] # remove case_name & metric
                            if data['result'] is None:
                                remove_none_list.append(name)
                        for name in remove_none_list:
                            case_results.pop(name)
                        self._suites_result[suite_name][arch][case_name]['results'] = case_results

    def get_suites_result(self):
        return self._suites_result

    def save_results_as_json(self, costomized_dir = None):
        file_path = os.path.join(self._file_path, 'results.json')
        if costomized_dir != None:
            file_path = os.path.join(costomized_dir, 'results.json')
        with open(file_path, 'w') as f:
            print(dump2json(self._suites_result), file=f)

    def print_info(self):
        # remove 'results' in self._suites_result, than print
        info_dict = deepcopy(self._suites_result)
        for suite_name in info_dict:
            for arch in info_dict[suite_name]:
                for case in info_dict[suite_name][arch]:
                    info_dict[suite_name][arch][case].pop('results')
        print(dump2json(info_dict))


if __name__ == '__main__':
    results = ResultsBuilder()
    results.print_info()
    results.save_results_as_json()
