from itertools import permutations, combinations
from collections import namedtuple
from typing import Iterable

NRE_LNG = namedtuple('NRE_LNG', ['num_required_elems', 'length'])


def generate_num_required_elems_to_required_len_map(max_unique_elems=5):
    class InvalidPattern(Exception):
        pass

    m = {}
    m[(0, 0)] = ()

    def _gen_impl(num_required_elems: int, required_len: int):
        mkey = NRE_LNG(num_required_elems, required_len)
        try:
            return m[mkey]
        except KeyError:
            pass
        if num_required_elems > required_len:
            raise InvalidPattern(f'{num_required_elems} {required_len}')
        if num_required_elems == 0:
            if required_len == 0:
                return []
            raise InvalidPattern(f'{num_required_elems} {required_len}')
        if num_required_elems == 1:
            if required_len > 0:
                m[mkey] = ((required_len,),)
                return m[mkey]
            raise InvalidPattern(f'{num_required_elems} {required_len}')

        res = []
        for n in range(1, required_len + 1):
            try:
                cur = _gen_impl(num_required_elems - 1, required_len - n)
                res += [(n,) + t for t in cur]
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


def generate_seed_patterns(acc, nrel_vals):
    res = []
    for val in nrel_vals:
        assert len(acc) == len(val)
        seed = []
        for char, vi in zip(acc, val):
            seed += [char, ] * vi
        res.append(tuple(seed))
    return res


class SwizzleGenerator:
    def __init__(self, max_unique_elems=4):
        self._nrel_map = generate_num_required_elems_to_required_len_map(
            max_unique_elems)

    def generate(self, accessors: Iterable[str], required_length: int):
        res = [self._gen_for_length(accessors, l)
               for l in range(1, required_length + 1)]
        return res

    def _gen_for_length(self, accessors, required_length):
        acc_list = list(accessors)
        res = []
        for l in range(1, required_length + 1):
            cur_len_patterns = set()
            for subacc in combinations(acc_list, l):
                nrel_key = NRE_LNG(l, required_length)
                nrel_vals = self._nrel_map[nrel_key]
                seed_patterns = generate_seed_patterns(subacc, nrel_vals)
                for sp in seed_patterns:
                    for p in permutations(sp):
                        cur_len_patterns.add(p)
            res += sorted(list(cur_len_patterns))
        return res


__all__ = [
    'SwizzleGenerator',
]

if __name__ == '__main__':
    sg = SwizzleGenerator()
    pats_per_len = sg.generate('xyz', 4)
    total = 0
    for i, pats in enumerate(pats_per_len):
        print(f'patterns at len={i}')
        for p in pats:
            print(f'  {p}')
        total += len(pats)
    # https://jojendersie.de/performance-optimal-vector-swizzling-in-c/
    print(f'total_patterns={total}')
