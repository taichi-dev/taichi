import pyoctnet
import numpy as np
import argparse
import os

def main(path, res, channels):
    pad = 2
    n_threads = 1
    R = np.identity(3, dtype = np.float32)

    grid = pyoctnet.Octree.create_from_off(path, res,res,res, R, pad=pad, n_threads=n_threads)
    dense = grid.to_cdhw()
    dense = np.tile(np.reshape(dense, (1, 1, res, res, res)), (1, channels, 1, 1, 1))
    print('Size: {}x{}x{}x{}'.format(channels, res, res, res))
    out_path = os.path.splitext(os.path.basename(path))[0] + '.bin'
    pyoctnet.write_dense(out_path, dense)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_paths', type = str, nargs = '+')
    parser.add_argument('-r', '--res', type = int, default = 256)
    parser.add_argument('-c', '--channels', type = int, default = 16)

    args = parser.parse_args()
    for m in args.model_paths:
        main(m, args.res, args.channels)
