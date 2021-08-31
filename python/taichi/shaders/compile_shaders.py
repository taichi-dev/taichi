from os import system, walk

files = []
for (dirpath, dirnames, filenames) in walk("."):
    files.extend(filenames)
    break

for f in files:
    if f[-5:] == ".frag" or f[-5:] == ".vert":
        spv_file = f[:-5] + "_" + f[-4:] + ".spv"
        system("glslc {} --target-env=vulkan -o {}".format(f, spv_file))