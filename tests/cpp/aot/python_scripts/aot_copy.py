import argparse
import json
import os
import shutil
import zipfile

import taichi as ti


def zip_files(srcDir):
    dstdir = srcDir + ".zip"
    z = zipfile.ZipFile(dstdir, 'w', zipfile.ZIP_DEFLATED)
    for dirpath, dirnames, filenames in os.walk(srcDir):
        fpath = dirpath.replace(srcDir, '')
        fpath = fpath and fpath + os.sep or ''
        for filename in filenames:
            z.write(os.path.join(dirpath, filename), fpath + filename)

    z.close()


def compile_graph_aot(arch):
    ti.init(arch=arch)

    if ti.lang.impl.current_cfg().arch != arch:
        return

    @ti.kernel
    def run0(base: int, arr: ti.types.ndarray(ndim=1, dtype=ti.i32)):
        for i in arr:
            arr[i] += base + i

    @ti.kernel
    def run1(base: int, arr: ti.types.ndarray(ndim=1, dtype=ti.i32)):
        for i in arr:
            arr[i] += base + i

    @ti.kernel
    def run2(base: int, arr: ti.types.ndarray(ndim=1, dtype=ti.i32)):
        for i in arr:
            arr[i] += base + i

    arr = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                       'arr',
                       ti.i32,
                       field_dim=1,
                       element_shape=(1, ))

    base0 = ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'base0', ti.i32)

    base1 = ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'base2', ti.i32)

    base2 = ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'base1', ti.i32)

    g_builder = ti.graph.GraphBuilder()

    g_builder.dispatch(run0, base0, arr)
    g_builder.dispatch(run1, base1, arr)
    g_builder.dispatch(run2, base2, arr)

    run_graph = g_builder.compile()

    assert "TAICHI_AOT_FOLDER_PATH" in os.environ.keys()
    pathdir = str(os.environ["TAICHI_AOT_FOLDER_PATH"])
    tmpdir = pathdir + "/compat-module.tcm"

    mod = ti.aot.Module()
    mod.add_graph('run_graph', run_graph)
    mod.archive(tmpdir)
    os.rename(tmpdir, pathdir + "/compat-module.zip")
    zip_file = zipfile.ZipFile(pathdir + "/compat-module.zip")
    zip_extract = zip_file.extractall(pathdir + "/compat_module")
    zip_file.close()
    list1 = []
    with open(pathdir + "/compat_module/metadata.json", 'r',
              encoding='utf8') as fp:
        json_data = json.load(fp)
        json_data.pop("required_caps")
        list1.append(json_data)
    with open(pathdir + "/compat_module/metadata.json", "w",
              encoding="UTF-8") as e:
        json_new_data = json.dumps(list1, ensure_ascii=False, indent=4)
        e.write(json_new_data[1:len(json_new_data)])
    os.remove(pathdir + "/compat-module.zip")
    zip_files(pathdir + '/compat_module')
    shutil.rmtree(pathdir + '/compat_module')
    os.rename(pathdir + "/compat_module.zip", tmpdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str)
    args = parser.parse_args()

    if args.arch == "cpu":
        compile_graph_aot(arch=ti.cpu)
    elif args.arch == "cuda":
        compile_graph_aot(arch=ti.cuda)
    elif args.arch == "vulkan":
        compile_graph_aot(arch=ti.vulkan)
    elif args.arch == "opengl":
        compile_graph_aot(arch=ti.opengl)
    else:
        assert False
