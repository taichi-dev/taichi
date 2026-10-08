# IB-WMLES PTX regression

This optional integration test compiles the real Simfinity transport kernels from
local `development` commit `d0e815c559afde3c41e16ecdd435f8b1b137bd68`.
It archives that commit from a supplied repository; it never checks out or edits
the user's solver working tree.

The compiler patch is `2026_10_taichi_compiler_v3.patch` under
`apps/solver/dev_scripts/sbarrett/unikernel_performance` at Simfinity commit
`4fc1b229d5f40ee48932c0d56cc4958b59c747e3`. Its SHA-256 is
`1f03d1c8e52b436ae5a3ab7ae4ccced50631d56be473f63eab66ee1c3c255b76`.

The configuration comes from
`apps/solver/dev_scripts/sbarrett/ONERA_wing_slater_example/run_onera_m6_wing.py`:
conservative TENO8A/HLLC advection with cell fallback, second-order viscosity with
the Debroeyer immersed wall model, and second-order heat conduction. The three
operators are **unfused kernel groups within one shared RK3 integration**. Cell
ownership, fused directions, fused RK sums, fused timestep collection, single
precision, and the normal compiler optimization settings are preserved. Only
the last group applies the RK sum, as in the production schedule.

A 16³, one-level octree with block size 8 and an immersed sphere replaces the large
wing mesh to reduce setup cost. Geometry values are runtime inputs. This is a
compiler regression, not a physical validation or a fused-kernel speed benchmark.
Each operator runs in a fresh process without offline caching. The fixture uses
the production group's dispatch function and compilation-options context to
materialize the main RK transport specializations, including the final timestep
reduction, and compares the entire PTX bytes, including symbols and directives.
It does not normalize instructions, registers, labels, or names. Setup, boundary,
and recovery-only kernels are outside this gold comparison.
Advection and viscosity each have two specializations (the first two RK stages
reuse one kernel); heat has three, for seven gold modules in total.
The internal `Program.materialize_cuda_kernel` binding runs the normal PTX JIT
and module-loading path but never launches the target kernel. Setup kernels still
run to construct the solver's real grid and immersed-wall resources. There is no
timestep execution, numerical output comparison, or runtime benchmark.

Use a Python 3.13 environment with the solver's locked dependencies and pytest.
Select the Taichi package built from this checkout through `PYTHONPATH` (including
its native extension and matching runtime bitcode). CUDA is required; CPU fallback
is disabled. Gold is specific to the LLVM toolchain and PTX target recorded in the
files; a different toolchain or target should fail rather than silently skip.

```sh
PYTHONPATH="$PWD/python" uv run --no-sync --project ../simfinity-mono/apps/solver python -m pytest \
  tests/ptx -v --simfinity-repo ../simfinity-mono --ptx-mode optimized
```

`optimized` enables all four patch flags: compact reaching definitions, compact
liveness, precomputed kill masks, and indexed CSE users. The worker checks that
the loaded native library actually contains these switches. `reference` disables
them, while `verify` additionally enables the patch's internal differential
checks. Verification reruns the reference CFG analyses and is not a performance
measurement.

Before applying a compiler change, explicitly record gold with an **unpatched**
native build:

```sh
PYTHONPATH="$PWD/python" uv run --no-sync --project ../simfinity-mono/apps/solver python -m pytest \
  tests/ptx -v --simfinity-repo ../simfinity-mono \
  --ptx-mode reference --record-ptx-gold
```

Then repeat without `--record-ptx-gold` to check baseline reproducibility, rebuild
with the compiler patch, and run `--ptx-mode optimized`. Recording refuses a
binary containing the patch. Gold PTX is losslessly gzip-compressed with a fixed
timestamp; manifests record hashes, native library provenance, compilation times,
and peak process RSS. Pytest's temporary directories retain the generated PTX,
`result.json`, and `compile.log` for failure diagnosis and timing comparisons.

Without `--simfinity-repo` (or `SIMFINITY_MONO`), these integration tests skip.

## Validation: 2026-10-08

All seven PTX modules were byte-identical with all four optimizations enabled:
three compile-only pytest cases passed. The 75 focused native compiler tests also
passed before and after the patch. A fresh unpatched heat compilation matched its
gold through the materialization-only API before applying the optimization patch.

LLVM 15.0.4, PTX target `sm_86`, Python 3.13.15:

| Operator | PTX modules | Baseline native compile (s) | Patched native compile (s) |
| --- | ---: | ---: | ---: |
| Advection | 2 | 368 | 116 |
| Viscosity | 2 | 61.1 | 12.4 |
| Heat conduction | 3 | 3.77 | 2.02 |
| Total | 7 | 433 | 130 |

These are single cold-process measurements of `Program.compile_kernel`; they
exclude solver setup, Python AST expansion, and PTX materialization. The observed
total native compilation time decreased by 69.9% (3.32x faster).

The patched native library SHA-256 was
`3163dc5fb76976c409cce6752c34e2690d07a365577378854fdbe85151508116`.
The advection gold was captured using the original first-launch API; viscosity
and heat gold, and all final patched comparisons, used materialization only.
