# Compiler compilation-time improvements

The objective is to reduce compilation cost without changing generated PTX.
The implemented optimizations change compiler data structures and lookup work;
they retain the existing optimization passes, transfer equations and alias rules.

## Compact dataflow sets

`taichi/ir/reaching_definition_set.h` assigns each relevant statement an integer
index within one analysis. Each control-flow node stores reaching definitions
and live variables in a vector of 64-bit words instead of a hash set of pointers.
Union becomes wordwise OR and set difference becomes AND with a complemented
mask. Iteration visits the set bits and maps their indices back to statements.

The motivation is to remove duplicated hash-table entries, pointer chasing and
per-entry allocations across nodes. Bulk set operations process 64 possible
members at once. The representation remains dense, so storage and bulk operation
cost scale with the size of the analysis universe, even when a set is sparse.

`TI_CFG_COMPACT_REACHING=1` and `TI_CFG_COMPACT_LIVE=1` independently enable the
two analyses. Their original hash-set implementations remain available.

## Precomputed kill masks

`taichi/ir/control_flow_graph.cpp` computes the definitions overwritten by each
node once, before the worklist iteration. The transfer equations remain:

```
reaching: OUT = GEN union (IN minus KILL)
liveness:  IN = GEN union (OUT minus KILL)
```

Previously, each visit repeated store-destination extraction and alias checks
for incoming definitions. With `TI_CFG_PRECOMPUTED_KILLS=1` and compact sets,
these checks produce a bit mask once and subsequent visits use word operations.
An address-to-definition index accelerates mask construction for scalar local
allocations and autodiff stack allocations. Other addresses retain the original
alias predicate. A multi-destination definition is killed only when all its
destinations are killed, preserving the original rule.

## Indexed users for common-subexpression elimination

`taichi/transforms/whole_kernel_cse.cpp` builds a reverse mapping from a statement
to statements that use it. Eliminating a duplicate expression can then update
its users directly, rather than repeatedly traversing the IR to locate users,
invalidate their visited state and replace operands.

`TI_CSE_INDEXED_USERS=1` enables this path. The index preserves the original
replacement scope restrictions and is rebuilt at the start of each CSE sweep.
Structural transformations that invalidate the index fall back to the original
traversal for the remainder of that sweep. This avoids using stale statement
pointers or omitting affected users.

## Indexed scalar-local store forwarding

The follow-up in `taichi/ir/control_flow_graph.cpp` is enabled by
`TI_CFG_INDEXED_FORWARDING=1`. Profiling the original patch put this work first:
in the first advection specialization's pre-offload simplification, forwarding
took 13.624 seconds, compared with 9.457 seconds for CSE. Reaching-definition
analysis accounted for only 0.070 seconds inside forwarding. This made reducing
forwarding searches a more promising target than further tuning propagation.
These figures describe one profiled run, not isolated speedup measurements.

The implementation keeps the latest retained store to each scalar local
allocation while walking a control-flow node. A load or store querying the same
allocation can obtain its previous value directly, instead of scanning every
preceding instruction. The index includes allocation's implicit zero definition.
Only the previous retained statement is inserted, after its simplification, so
an erased redundant store cannot become a dangling forwarding candidate. Values
are read from the retained store on demand, reflecting operand replacements.

For cross-block queries, reaching and generated definitions are bucketed by
scalar allocation. Searches visit only the relevant bucket. Each bucket retains
the original set's iteration order, preserving the choice between equivalent
visible values. Matrix-pointer origins are included conservatively, and the
original alias, value-equality and visibility checks still run on candidates.
The index is rebuilt for each node traversal; it is not retained across passes.

This fast path is limited to non-tensor, non-quantized local allocations.
Other addresses use the original search. `TI_CFG_VERIFY_FORWARDING=1` compares
every lookup's returned statement pointer with the original search before the
IR is changed. Native regression cases cover erased stores and rewritten values,
conflicting/equal definitions at branch merges, and tensor-element fallback.

## Validation and performance evidence

`tests/ptx` compiles advection, viscosity and heat conduction as separate operator
groups within the production shared RK3 schedule. The solver source is pinned to
local development commit `d0e815c559afde3c41e16ecdd435f8b1b137bd68`.
Seven complete PTX modules are compared byte-for-byte with immutable gold from
the unpatched compiler. Target kernels are materialized but not launched.
See `tests/ptx/README.md` for configuration, toolchain and reproduction commands.

The original patch is committed as `96bbee769`; its regression tests and
compile-only binding are committed separately as `3cbe7e02a`.

Initial measurements on 2026-10-08, LLVM 15.0.4 and target sm_86:

| Operator | Original native compile (s) | Patched native compile (s) | Speedup |
| --- | ---: | ---: | ---: |
| Advection | 368.151 | 115.905 | 3.18x |
| Viscosity | 61.128 | 12.397 | 4.93x |
| Heat conduction | 3.773 | 2.022 | 1.87x |
| Total | 433.052 | 130.324 | 3.32x |

All seven modules matched, and 75 focused native compiler tests passed before
and after the patch. These are single cold-process measurements with offline
caching disabled. Timings cover `Program.compile_kernel`, excluding solver setup,
Python AST expansion and PTX materialization. They do not isolate the individual
optimizations' contributions or establish performance on operator-fused kernels.

The optional `TI_CFG_VERIFY_REACHING` and `TI_CFG_VERIFY_LIVE` switches rerun the
original analyses and compare each node's input/output sets. `TI_CSE_VERIFY_USERS`
compares indexed users with a fresh traversal. Verification adds work and must
be disabled for timing comparisons. The offline-cache key includes optimization
and verification switches so experimental modes cannot share compiled entries.

Pass timings can be captured with `pytest tests/ptx --ptx-profile` alongside the
usual repository and mode options. Each target native compilation clears and
prints the scoped profiler; the output is retained in its `compile.log`.

`tests/ptx/benchmark.py` compares two built packages using repeated fresh-process
runs in alternating order. Every run checks the same seven gold modules; no gold
is regenerated. It records individual native compilation times, library hashes
and peak process RSS, then reports median times and ranges. Differential checks
are run separately to avoid including their overhead in performance results.

The forwarding follow-up passed all 78 focused native tests with verification
enabled (the original 75 plus three new regression cases). A separate real-kernel
run with `--ptx-verify-forwarding` passed all three operator tests: every indexed
lookup matched the original lookup, and all seven PTX modules matched gold.
The new tests also passed with indexing disabled.

## Forwarding follow-up measurements: 2026-10-08

Two fresh-process samples per build, in baseline/candidate/candidate/baseline
order, compared the original patch (`96bbee769`) with the additional forwarding
index. All four samples passed all seven PTX comparisons. Verification and
profile printing were disabled for these measurements.

| Operator | Original patch median (s) | With forwarding index median (s) | Speedup |
| --- | ---: | ---: | ---: |
| Advection | 97.164 | 83.890 | 1.16x |
| Viscosity | 11.522 | 10.778 | 1.07x |
| Heat conduction | 1.930 | 2.048 | 0.94x |
| Total | 110.616 | 96.716 | 1.14x |

The combined native compilation time decreased by **12.6%**, an additional
improvement over the original patch. Control totals ranged from 105.825 to
115.407 seconds; candidate totals ranged from 96.632 to 96.800 seconds. With only
two samples per build, these ranges are descriptive, not confidence intervals.
The measurements cover the combined intra-block and cross-block indexes, not
their separate contributions. Do not multiply this ratio by the earlier 3.32x
result: those original/unpatched measurements were taken in a separate run.

Heat conduction regressed by 0.118 seconds (6.1%) at the median: constructing the
indexes adds work, and the smaller operator offers less opportunity to amortize
it. This is an aggregate improvement, not a speedup of every operator.

Peak process RSS includes solver setup and PTX materialization as well as native
compilation. Advection ranged from 939.9–985.1 MiB for the control and
981.0–983.4 MiB for the candidate; these samples do not establish a consistent
memory improvement. Viscosity ranged from 725.1–735.0 versus 728.8–733.4 MiB,
and heat from 636.9–638.0 versus 637.2–637.6 MiB.

Individual samples, summary statistics and native-library hashes are retained in
[`tests/ptx/measurements/forwarding-2026-10-08.json`](tests/ptx/measurements/forwarding-2026-10-08.json).
The test and benchmark additions are committed separately as `0bee6b05e`.

## Repairing CSE users after branch hoisting

The original indexed CSE path invalidated its user map after hoisting common
statements out of an `if`. Every later elimination in that sweep then reverted
to whole-IR searches. `TI_CSE_LOCAL_REPAIR=1` maintains the index across hoists,
preserving the reference traversal's view of the IR, including excluding
extracted statements awaiting delayed insertion. Before hoisting, it removes
operand edges belonging to the extracted true-branch subtree and the affected
false branch. After the original replacement and erasure operations, it adds
the false branch's current edges back. Users elsewhere in the IR are preserved.
This avoids rebuilding the entire graph after each hoist. The full index is
still rebuilt at the start of each CSE sweep. The superseded whole-index repair
prototype was removed; only local repair remains in the compiler.

In independent screening, whole-index repair reduced aggregate scoped CSE time
from 22.771 to 5.119 seconds. With the AST improvement below enabled, local repair
then reduced CSE from 5.923 to 3.894 seconds; subtree maintenance itself took
0.007 seconds. All screening PTX comparisons passed. These are individual pass
measurements, not repeated end-to-end speedup estimates.

## Avoiding unused AST replacement scans

`TI_AST_SKIP_UNUSED_REPLACE=1` maintains a conservative set of statements used
as operands while lowering the frontend AST. Most replaced frontend statements
have no registered statement users, but previously each replacement still
traversed the IR looking for them. The new path skips that traversal when the
old statement is absent from the referenced set.

The set is initialized from the entire input IR and updated with operands from
newly lowered statements, including their subtrees. Replacement targets are
marked as referenced when existing users are redirected to them. Entries are
never removed: stale entries can only cause an unnecessary original traversal,
not a missed replacement. The existing replacement scope and ordering are
unchanged when users may exist. A native regression deliberately mixes frontend
allocation and existing backend users to exercise that fallback.

Independent screening reduced scoped AST-lowering time from 24.923 to 1.167
seconds, with all seven PTX modules unchanged. Unlike removing optimization
passes, this removes searches that cannot change any operands.

The full experiment record, including rejected approaches and preserved
prototype patches, is in [`tests/ptx/experiments/README.md`](tests/ptx/experiments/README.md).

## Lazy and shared forwarding indexes

Profiling exposed a cost hidden inside store forwarding: building each node's
address index took 18.495 seconds in one CSE/AST-optimized run.
`TI_CFG_LAZY_FORWARDING_INDEX=1` defers construction until the first scalar-local
cross-block query. Queries satisfied by the latest store in the same node never
need it. This reduced index construction to 15.239 seconds and the sampled total
from 48.916 to 45.807 seconds, preserving all PTX bytes.

`TI_CFG_SHARED_FORWARDING_INDEX=1` removes most of the remaining duplication.
The graph builds one address index from the compact reaching-definition
universe. Each node filters its address bucket through its own `reach_in`
membership test, instead of constructing another copy of the index. The bucket
order is the universe's order, exactly matching compact-set iteration, so
filtering preserves which equivalent visible definition is selected first.
Generated definitions remain indexed locally in their original order.

Sharing is restricted to compact reaching sets. When the analysis uses hash
sets, including after differential reaching-analysis verification, it falls back
to the original per-node index to preserve that representation's iteration order.
The shared index lasts for one forwarding pass. It indexes scalar allocation
identities; stored values are still read from the live statements at query time,
and the original alias, equality and visibility predicates remain in use.

In screening on top of local CSE repair, AST pruning and lazy construction,
sharing reduced index construction from 14.623 seconds to 0.041 seconds
(0.020 shared plus 0.021 local). Total forwarding time fell from 20.769 to 4.580
seconds, and native compilation from 45.927 to 29.433 seconds. All seven PTX
modules remained byte-identical. Repeated final measurements are recorded below.

## Final repeated comparison (2026-10-08)

The final comparison uses one native build, switching the new optimizations off
for the control and on for the candidate. Both retain the original compact CFG,
KILL-mask, indexed CSE and scalar-forwarding improvements. The candidate adds
local CSE repair, AST replacement pruning, and lazy/shared forwarding indexes.
Each variant runs twice in fresh processes, in control/candidate/candidate/control
order, with offline caching, profiling and differential verification disabled.
Times cover native `Program.compile_kernel` for the operator-unfused kernels.

| Operator | Control median (s) | Candidate median (s) | Speedup |
| --- | ---: | ---: | ---: |
| Advection | 87.142 | 26.505 | 3.29× |
| Viscosity | 11.602 | 3.725 | 3.12× |
| Heat conduction | 2.072 | 0.841 | 2.46× |
| Total | 100.816 | 31.071 | 3.24× |

This is a 69.2% reduction in compilation time relative to the compiler at the
start of this experiment round. Control totals ranged from 95.669 to 105.964
seconds; candidate totals ranged from 30.654 to 31.487 seconds. Two repetitions
establish the large improvement but are insufficient for a precise noise model.
Do not multiply this ratio by earlier measurements made under different conditions.
All four runs matched all seven original PTX modules byte for byte.

The [measurement record](tests/ptx/measurements/compiler-final-2026-10-08.json)
includes individual kernel timings, compiler flags, native-library hashes and
peak process RSS. The improvements remain opt-in; reproduce the candidate with
`--ptx-mode optimized --ptx-experiment cse-local --ptx-experiment ast-unused
--ptx-experiment lazy-forwarding --ptx-experiment shared-forwarding`.

Final validation also passed all three operator tests with forwarding differential
verification enabled (seven unchanged PTX modules). This used optimized mode plus
`--ptx-verify-forwarding`, so compact reaching sets and the shared index remained
active. The broader verify mode rebuilds reaching sets in the reference hash
representation and therefore exercises the per-node fallback instead. Separately,
82 native tests passed with the retained features and forwarding, liveness and
CSE-user checks enabled; all seven new native tests also passed with the
optimizations disabled. No target transport kernels were numerically executed.

## Pre-all-patches versus retained-all-patches A/B

The final cleanup removes the superseded whole-index CSE repair path and its
cache switch. Unsuccessful prototypes were already absent from compiler source;
their archived patches and measurements remain solely as an experiment record.
The retained compiler passed all 82 focused native tests again after cleanup.

The baseline is the preserved native build of `ba0e81dce559fb63a5958bf82feb1d00c55c02fe`
with only the compile-only test binding, before any compiler optimization patches.
Its SHA-256 is `9d936b73e522c5a3a11abd6c910281389ce82fc1675d52ed66e1de49c6415925`,
matching the compile-only gold provenance. The candidate is the rebuilt compiler
with all retained optimizations enabled. Both builds use LLVM 15.0.4, Clang 15,
Release configuration, Python 3.13.15, target sm_86 and identical runtime bitcode.
The solver remains pinned to the same development commit and operator-unfused
fixture. No gold was regenerated.

The A/B comparison runs serially in baseline/candidate/candidate/baseline order,
with two fresh-process samples per build. Offline caching, scoped profiling and
differential verification are disabled for timing. Each sample validates all
seven complete PTX modules. Timings measure native compilation only, excluding
solver setup, Python AST expansion and PTX materialization. Target transport
kernels are not launched.

Reproduce with the solver's Python environment and separately built packages:

```sh
python tests/ptx/benchmark.py \
  --baseline-pythonpath /path/to/unpatched/python --baseline-mode reference \
  --candidate-pythonpath "$PWD/python" --candidate-mode optimized \
  --candidate-experiment cse-local --candidate-experiment ast-unused \
  --candidate-experiment lazy-forwarding --candidate-experiment shared-forwarding \
  --simfinity-repo ../simfinity-mono \
  --output /tmp/ib-wmles-all-patches-ab --repeats 2
```

| Operator | Pre-all-patches median (s) | Retained-all-patches median (s) | Speedup |
| --- | ---: | ---: | ---: |
| Advection | 269.170 | 26.227 | 10.26× |
| Viscosity | 60.047 | 3.471 | 17.30× |
| Heat conduction | 3.909 | 0.762 | 5.13× |
| Total | 333.126 | 30.460 | 10.94× |

This directly measured comparison gives **10.94× faster compilation (90.9% less
time)** for the complete retained set. Baseline totals were 335.229 and 331.022
seconds; candidate totals were 30.740 and 30.179 seconds. Every sample passed
all three tests and all seven byte-exact PTX comparisons. Two repetitions per
build establish the large effect but do not provide a precise noise model or
evidence for operator-fused kernels. This result does not multiply ratios from
the earlier experiment rounds.

[Individual samples, flags, library hashes and summary statistics](tests/ptx/measurements/all-patches-ab-2026-10-08.json)
are preserved for review.

## Fully fused compilation timing

With the retained compiler (`af45e5b4f`), the same pinned solver development
configuration also compiles with advection, viscosity and heat conduction in one
kernel group. Directions, RK sums and timestep collection remain fused. The RK3
schedule materializes three distinct variants. Two fresh-process measurements,
with offline caching and verification disabled, gave:

| Fused variant | Median native compile time (s) |
| --- | ---: |
| Base update | 21.582 |
| Update with RK sum | 22.485 |
| Update with RK sum and timestep | 24.243 |
| Total | 68.309 |

Native totals were 71.693 and 64.926 seconds. Complete process wall times were
108.619 and 93.586 seconds (median 101.103 seconds), including setup, Python AST
expansion and PTX materialization. The native total is about 2.24 times the earlier
30.460-second operator-unfused total; this is a comparison of separate measurement
runs, not an interleaved fusion A/B experiment. No target transport kernels ran.
All three fused PTX modules were byte-identical between repetitions. This is
repeatability evidence, not a pre/post-patch fused PTX equivalence check.

[Full fused measurement records](tests/ptx/measurements/fused-2026-10-08.json)
include individual timings, PTX hashes, feature switches and library provenance.
The compile worker now accepts `--method fused` for this measurement.
