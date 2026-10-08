# Compiler experiment record

These experiments use the operator-unfused IB-WMLES fixture, pinned solver
development sources, and the original seven PTX gold modules. No gold was
regenerated. Every completed screening run below passed all seven comparisons.
Times measure native `Program.compile_kernel` calls, excluding setup, Python
AST expansion, and PTX materialization. Profiling was enabled for screening.

## Independent screening

The control already includes the original compact-CFG/indexed-CSE patch and
scalar-local forwarding (`357325e72`). Runs were serial, with controls at both
ends. Individual candidate totals are single samples, not reliable speedup
estimates on their own. Scoped pass timings help distinguish the intended
effect from timing variation elsewhere in the compiler.

| Variant | Native total (s) | Relevant scoped measurement | Decision |
| --- | ---: | --- | --- |
| Opening control | 90.503 | AST 24.923 s; CSE 22.771 s; reaching + liveness 0.519 s | Reference |
| Repair CSE index after hoisting | 73.640 | CSE 5.119 s | Superseded by local repair |
| CSE buckets by SNode | 93.294 | CSE 23.561 s | Not retained |
| Skip unused AST replacement scans | 67.867 | AST 1.167 s | Retained |
| Precomputed GEN and reusable transfer buffers | 95.667 | Reaching + liveness 0.518 s | Not retained |
| Dense worklist membership and mask lookup | 91.293 | Reaching + liveness 0.519 s | Not retained |
| Reverse-postorder priority worklist | 88.440 | Reaching + liveness 0.511 s | Not retained |
| Closing control | 91.901 | AST 25.193 s; CSE 23.204 s; reaching + liveness 0.520 s | Reference |

The dataflow variants did not establish a material compilation improvement on
this workload. In particular, the reverse-postorder run's lower total should not
be attributed to its approximately 8 ms difference in dataflow time. The other
passes varied by much more. These results do not rule out benefits on different
graphs or operator-fused kernels.

## Follow-ups suggested by profiling

The successful CSE/AST combination was the control for this round. Local CSE
repair removes and reconstructs operand edges only for affected subtrees. The
store-candidate experiment skips instructions without store destinations during
forwarding searches, retaining ordering, alias checks and the block prefix
needed by cross-node searches.

| Variant | Native total (s) | Relevant scoped measurement | Decision |
| --- | ---: | --- | --- |
| Whole-index CSE repair + AST | 51.214 | CSE 5.923 s; forwarding 25.194 s | Reference |
| Local CSE repair + AST | 49.153 | CSE 3.894 s; local repair 0.007 s | Retained |
| Local CSE + AST + store-only candidates | 50.795 | Forwarding 27.075 s | Store-only lists not retained |
| Add SNode buckets to that combination | 51.836 | No improvement in combined compilation | SNode buckets not retained |
| Closing whole-index CSE repair + AST | 51.531 | Reference repeat | Reference |

Raw operator timings, scoped measurements, switches and native-library hashes:

- [`compiler-screening-2026-10-08.json`](../measurements/compiler-screening-2026-10-08.json)
- [`compiler-followups-2026-10-08.json`](../measurements/compiler-followups-2026-10-08.json)

## Reproducing retired prototypes

`screening.patch.gz` and `followups.patch.gz` preserve the corresponding compiler
source changes relative to `357325e72`. Apply one to that base in a separate
checkout and build it with the same LLVM toolchain. Use the current `tests/ptx`
harness, which retains the historical experiment names; the worker checks that
the loaded native library actually supports every requested switch.

For example, `--ptx-experiment gen-transfer` selects the archived GEN/transfer
prototype. Requesting a retired experiment against the current compiler fails
explicitly instead of silently measuring the control. `sweep.py --variants`
accepts individual names or combinations separated by `+`. It retains failures
and continues testing subsequent variants.

`benchmark.py` accepts repeated `--baseline-experiment` and
`--candidate-experiment` options. The two package paths may be identical when
comparing switches in one build; this avoids rebuild differences. Repeated
comparisons alternate execution order and leave verification disabled.

## Unknown-input forwarding shortcut

After an intra-block search fails, an incoming definition for the exact queried
address with no stored value guarantees that the reference cross-block search
will reject forwarding. An early membership check therefore preserves the
result. Testing it with local CSE repair and AST pruning passed all PTX checks,
but increased the sampled total from 48.114 to 48.998 seconds; forwarding stayed
at 24.532 versus 24.922 seconds. It was not retained.

[`compiler-unknown-input-2026-10-08.json`](../measurements/compiler-unknown-input-2026-10-08.json)
contains the records; `unknown-input.patch.gz` preserves the compiler prototype.

## Forwarding-index construction

Profiling after the unsuccessful search shortcuts identified index construction
as the dominant forwarding cost. Two further variants were implemented and
tested. Each row below passed all seven PTX comparisons.

| Comparison | Native total before → after (s) | Index construction before → after (s) | Decision |
| --- | ---: | ---: | --- |
| Eager → lazy per-node construction | 48.916 → 45.807 | 18.495 → 15.239 | Retained |
| Lazy per-node → shared universe index | 45.927 → 29.433 | 14.623 → 0.041 | Retained |

These are single screening comparisons with local CSE repair and AST pruning
enabled. Final repeated measurements compare the complete retained combination
with this round's original control. Details of the ordering and fallback rules
are in [`improvements.md`](../../../improvements.md).

- [`compiler-lazy-forwarding-2026-10-08.json`](../measurements/compiler-lazy-forwarding-2026-10-08.json)
- [`compiler-shared-forwarding-2026-10-08.json`](../measurements/compiler-shared-forwarding-2026-10-08.json)

This completes eleven experiments/variants: whole-index CSE repair, local CSE
repair, SNode CSE buckets, AST pruning, GEN/transfer reuse, dense worklists,
reverse-postorder worklists, store-only searches, unknown-input rejection, lazy
forwarding indexes and shared forwarding indexes.

## Final repeated result

Two fresh-process measurements per variant, in alternating order, gave median
native totals of **100.816 seconds for the control and 31.071 seconds for the
retained combination (3.24× faster, 69.2% less time)**. All seven gold PTX modules
matched in every run. These measurements exclude profiling and differential
verification; the control and candidate use the same native library with different
feature switches. See the [full samples](../measurements/compiler-final-2026-10-08.json)
and [per-operator table and caveats](../../../improvements.md#final-repeated-comparison-2026-10-08).

Final validation additionally passed all three PTX tests with forwarding
differential verification while compact/shared indexing remained enabled,
82 native tests with the retained features and differential checks, and seven
new native tests with optimizations disabled. The final measurement JSON includes
the separate forwarding-verification records.

The final retained-only compiler removes the superseded whole-index CSE repair
path as well as all unsuccessful prototypes. Historical switches remain in the
test harness solely for reproducing archived results; unsupported switches fail
against the current native build.

The subsequent [pre-all-patches A/B comparison](../measurements/all-patches-ab-2026-10-08.json)
measured 333.126 → 30.460 seconds (10.94×) after this cleanup, using two cold
runs per build. All seven PTX modules matched in all four samples; all 82 native
tests passed again. See [the methodology and per-operator results](../../../improvements.md#pre-all-patches-versus-retained-all-patches-ab).
