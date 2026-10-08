"""Compile one operator of development's ONERA IB-WMLES configuration.

Invoked in a fresh process by test_ib_wmles.py. The small immersed sphere and
single octree level limit setup cost; transport, wall model and RK policies come
from the pinned development case. Operator grouping is the only policy change.
"""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
from dataclasses import replace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-root", type=Path, required=True)
    parser.add_argument("--method", choices=("advection", "viscosity", "heat_conduction"), required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.solver_root / "src"))
    sys.path.insert(0, str(args.solver_root / "dev_scripts/sbarrett/ONERA_wing_slater_example"))

    import taichi as ti
    import trimesh
    from taichi._lib import core
    from taichi.lang import kernel_impl
    from simfinity_solver.immersed_boundary.triangulated_surface import (
        TriangulatedSurface,
    )
    from simfinity_solver.operators.integration_group import (
        AdvectionPhysicsMethod,
        ExecutionPolicy,
        Fusion,
        HeatConductionPhysicsMethod,
        IntegrationGroupOperatorFactory,
        LoopOwnership,
        MaximumReduction,
        TimeIntegratorKind,
        ViscosityPhysicsMethod,
    )
    from simfinity_solver.solver.solver_factory import SolverFactory, SolverOption
    import simfinity_solver

    assert Path(simfinity_solver.__file__).is_relative_to(args.solver_root / "src")
    library = Path(core.__file__)
    library_bytes = library.read_bytes()
    if os.environ.get("TI_CFG_COMPACT_REACHING") == "1":
        for flag in (
            b"TI_CFG_COMPACT_REACHING",
            b"TI_CFG_COMPACT_LIVE",
            b"TI_CFG_PRECOMPUTED_KILLS",
            b"TI_CSE_INDEXED_USERS",
        ):
            assert flag in library_bytes, f"Compiler does not implement {flag.decode()}: {library}"
    wing = importlib.import_module("run_onera_m6_wing")
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.3)
    mesh.apply_translation((0.0, 1.0, 0.0))
    mesh.export("sphere.stl")
    # TriangulatedSurface constructs Taichi resources, so initialise first.
    ti.init(
        arch=ti.cuda,
        enable_fallback=False,
        offline_cache=False,
        default_fp=ti.f32,
        print_kernel_asm=True,
        device_memory_GB=1,
    )
    config = replace(
        wing.make_config(TriangulatedSurface("sphere.stl"), wing.make_boundary_condition_config()),
        nx=16,
        ny=16,
        nz=16,
        n_levels=1,
        max_steps=1,
        x_min=-1.0,
        x_max=1.0,
        y_min=0.0,
        y_max=2.0,
        z_min=-1.0,
        z_max=1.0,
    )
    solver = SolverFactory.create(SolverOption.IMMERSED_NAVIER_STOKES, config)
    production = tuple(op.inner_operator.second_operator for op in solver.operators)
    preparer = solver.operators[0].inner_operator.first_in_place_operator
    methods = (
        AdvectionPhysicsMethod(config.advection_operator_config, production_operator=production[0]),
        ViscosityPhysicsMethod(config.dns_viscosity_operator_config, numerics=production[1]),
        HeatConductionPhysicsMethod(config.heat_conduction_operator_config, numerics=production[2]),
    )
    method = next(method for method in methods if method.name == args.method)
    group_index = methods.index(method)
    operator = IntegrationGroupOperatorFactory.create(
        methods,
        integrator=TimeIntegratorKind.RK3,
        execution_policy=ExecutionPolicy(
            loop_ownership=LoopOwnership.CELL,
            kernel_groups=tuple((item.name,) for item in methods),
            direction_fusion=Fusion.FUSED,
            timestep_fusion=Fusion.FUSED,
            rk_sum_fusion=Fusion.FUSED,
            maximum_reduction=MaximumReduction.GLOBAL_ATOMIC,
        ),
        grid_type=type(solver.grid),
        bc_applier=solver.bc_applier,
        input_preparer=preparer,
        operator_name="ptx_operator_unfused",
    )
    if args.method == "viscosity":
        assert operator.get_wall_model_sampler() is not None
    ti.sync()
    # Keep setup PTX separate from the transport modules under test.
    for path in Path.cwd().glob("taichi_kernel_nvptx_*.ptx"):
        path.unlink()

    original_create = core.Program.create_kernel
    original_transform = kernel_impl.transform_tree
    original_compile = core.Program.compile_kernel
    original_launch = core.Program.launch_kernel
    creating = None
    bodies = {}
    kernels = {}
    compiled = {}
    timings = {}
    records = []

    def create(program, callback, name, *extra):
        nonlocal creating
        creating = name
        try:
            kernel = original_create(program, callback, name, *extra)
        finally:
            creating = None
        kernels[id(kernel)] = name
        return kernel

    def transform(tree, ctx, *extra, **kwargs):
        body = ctx.global_vars.get("body")
        if creating and body is not None:
            bodies[creating] = getattr(body, "__qualname__", "")
        return original_transform(tree, ctx, *extra, **kwargs)

    def compile_kernel(program, cfg, caps, kernel):
        name = kernels.get(id(kernel))
        start = time.perf_counter()
        value = original_compile(program, cfg, caps, kernel)
        timings[name] = time.perf_counter() - start
        compiled[id(value)] = name
        return value

    def launch(program, data, ctx):
        name = compiled.get(id(data))
        body = bodies.get(name, "")
        before = set(Path.cwd().glob("taichi_kernel_nvptx_*.ptx"))
        if body.startswith("FluxLoopKernels._cell_body"):
            value = program.materialize_cuda_kernel(data)
        else:
            value = original_launch(program, data, ctx)
        paths = sorted(set(Path.cwd().glob("taichi_kernel_nvptx_*.ptx")) - before)
        if body.startswith("FluxLoopKernels._cell_body"):
            for path in paths:
                target = Path(f"{len(records):02d}_{body.split('.')[-1]}.ptx")
                path.rename(target)
                records.append(
                    {
                        "file": target.name,
                        "body": body,
                        "kernel": name,
                        "seconds": timings[name],
                        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    }
                )
                print("CAPTURED", records[-1], flush=True)
        return value

    core.Program.create_kernel = create
    kernel_impl.transform_tree = transform
    core.Program.compile_kernel = compile_kernel
    core.Program.launch_kernel = launch
    try:
        # Materialize the same production specs as the RK schedule, without
        # executing them or compiling fallback/recovery passes between stages.
        # Runtime state values do not affect these template specializations.
        with operator._compilation_options():
            operator._reset_inverse_timesteps()
            operator._prepare_input(solver.grid, 0.0, solver.grid.current, solver.material_properties, 1.0e-7)
            for stage, (reference_weight, euler_weight) in enumerate(((0.0, 1.0), (0.75, 0.25), (1 / 3, 2 / 3))):
                method.reset_update_status()
                method.configure_recovery(euler_weight=euler_weight, calculate_timestep=stage == 2)
                operator._apply_kernel_group(
                    solver.grid,
                    operator._update_specs[group_index],
                    operator._kernel_timestep_fields[group_index],
                    solver.grid.current,
                    solver.grid.scratch,
                    solver.material_properties,
                    1.0e-7,
                    calculate_timestep=stage == 2,
                    timestep_kernels=operator._update_with_timestep_specs[group_index],
                    apply_rk_sum=stage > 0 and group_index == len(methods) - 1,
                    rk_sum_kernels=operator._update_with_rk_sum_specs[group_index],
                    rk_sum_timestep_kernels=operator._update_with_rk_sum_and_timestep_specs[group_index],
                    reference_handle=solver.grid.current if stage > 0 else None,
                    reference_weight=reference_weight,
                    euler_weight=euler_weight,
                )
        ti.sync()
    finally:
        core.Program.create_kernel = original_create
        kernel_impl.transform_tree = original_transform
        core.Program.compile_kernel = original_compile
        core.Program.launch_kernel = original_launch
    expected = ["FluxLoopKernels._cell_body", "FluxLoopKernels._cell_body_with_timestep"]
    if group_index == len(methods) - 1:
        expected = [
            "FluxLoopKernels._cell_body",
            "FluxLoopKernels._cell_body_with_rk_sum",
            "FluxLoopKernels._cell_body_with_rk_sum_and_timestep",
        ]
    assert [item["body"] for item in records] == expected, records
    result = {
        "method": args.method,
        "compile_only": True,
        "records": records,
        "taichi_commit": core.get_commit_hash(),
        "taichi_library": str(library),
        "taichi_library_sha256": hashlib.sha256(library_bytes).hexdigest(),
        "compiler_has_patch": b"TI_CFG_COMPACT_REACHING" in library_bytes,
        "llvm_version": core.get_llvm_target_support(),
        "cuda_compute_capability": core.query_int64("cuda_compute_capability"),
        "python_version": platform.python_version(),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "compiler_flags": {key: value for key, value in os.environ.items() if key.startswith(("TI_CFG_", "TI_CSE_"))},
    }
    Path("result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
