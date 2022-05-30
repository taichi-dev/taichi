from collections import namedtuple
from itertools import combinations, permutations
from typing import Iterable, List, Tuple

NRE_LEN = namedtuple('NRE_LEN', ['num_required_elems', 'length'])


def generate_num_required_elems_required_len_map(max_unique_elems=5):
    '''
    For example, if we want a sequence of length 4 to be composed by {'x', 'y'},
    we have the following options:

    * 'xxxx': 4 'x', 0 'y'
    * 'xyyy': 1 'x', 3 'y'
    * 'xxyy': 2 'x', 2 'y'
    * 'xxxy': 3 'x', 1 'y'
    * 'yyyy': 0 'x', 4 'y'

    Each of these pattern is a seed. We can then do a permutation on it to get
    all the patterns for this seed.

    NRE_LEN(2, 4) maps to [(4, 0), (1, 3), (2, 2), (3, 1), (0, 4)]
    '''
    class InvalidPattern(Exception):
        pass

    m = {}
    m[(0, 0)] = ()

    def _gen_impl(num_required_elems: int, required_len: int):
        mkey = NRE_LEN(num_required_elems, required_len)
        try:
            return m[mkey]
        except KeyError:
            pass
        invalid_pat = InvalidPattern(f'{num_required_elems} {required_len}')
        if num_required_elems > required_len:
            raise invalid_pat
        if num_required_elems == 0:
            if required_len == 0:
                return []
            raise invalid_pat
        if num_required_elems == 1:
            if required_len > 0:
                m[mkey] = ((required_len, ), )
                return m[mkey]
            raise invalid_pat

        res = []
        for n in range(1, required_len + 1):
            try:
                cur = _gen_impl(num_required_elems - 1, required_len - n)
                res += [(n, ) + t for t in cur]
            except InvalidPattern:
                pass
        res = tuple(res)
        m[mkey] = res
        return res

    upperbound = max_unique_elems + 1
    for num_req in range(1, upperbound):
        for required_len in range(num_req, upperbound):
            _gen_impl(num_req, required_len)

    return m


class SwizzleGenerator:
    def __init__(self, max_unique_elems=4):
        self._nrel_map = generate_num_required_elems_required_len_map(
            max_unique_elems)

    def generate(self, accessors: Iterable[str],
                 required_length: int) -> List[Tuple[int, ...]]:
        res = []
        for l in range(required_length):
            res += self._gen_for_length(accessors, l + 1)
        return res

    def _gen_for_length(self, accessors, required_length):
        acc_list = list(accessors)
        res = []
        for num_required_elems in range(1, required_length + 1):
            cur_len_patterns = set()
            for subacc in combinations(acc_list, num_required_elems):
                nrel_key = NRE_LEN(num_required_elems, required_length)
                nrel_vals = self._nrel_map[nrel_key]
                seed_patterns = self._generate_seed_patterns(subacc, nrel_vals)
                for sp in seed_patterns:
                    for p in permutations(sp):
                        cur_len_patterns.add(p)
            res += sorted(list(cur_len_patterns))
        return res

    @staticmethod
    def _generate_seed_patterns(acc, nrel_vals):
        res = []
        for val in nrel_vals:
            assert len(acc) == len(val)
            seed = []
            for char, vi in zip(acc, val):
                seed += [char] * vi
            res.append(tuple(seed))
        return res


__all__ = [
    'SwizzleGenerator',
]
