import argparse
import os

import pytest

from tests import test_utils


@pytest.mark.skipif(os.environ.get('TI_LITE_TEST') or '0', reason='Lite test')
def test_simple_derivative():
    from taichi.examples.autodiff.simple_derivative import initialize

    initialize()


@pytest.mark.skipif(os.environ.get('TI_LITE_TEST') or '0', reason='Lite test')
def pic_simple_derivative(result_dir):
    from matplotlib import pyplot as plt
    from taichi.examples.autodiff.simple_derivative import (grad_xs,
                                                            initialize, xs, ys)

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
    test_utils.mkdir_p(result_dir)
    plt.savefig(result_dir + '/output.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate simple_derivative pic')
    parser.add_argument('output_directory',
                        help='output directory of generated pic')
    pic_simple_derivative(parser.parse_args().output_directory)
