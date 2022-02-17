import argparse


def mkdir_p(dir_path):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''
    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(dir_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(dir_path):
            pass
        else: raise


def pic_simple_derivative(result_dir):
    from taichi.examples.autodiff.simple_derivative import (initialize,
                                                            xs, ys, grad_xs)
    from matplotlib import pyplot as plt

    initialize()

    plt.title('Auto Diff')
    ax = plt.gca()
    ax.plot(xs, ys, label='f(x)')
    ax.plot(xs, grad_xs, label='f\'(x)')
    ax.legend()
    ax.grid(True)
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')

    # Create new directory
    mkdir_p(result_dir)
    plt.savefig(result_dir + '/output.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate simple_derivative pic')
    parser.add_argument('output_directory',
                        help='output directory of generated pic')
    pic_simple_derivative(parser.parse_args().output_directory)
