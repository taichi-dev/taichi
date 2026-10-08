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
