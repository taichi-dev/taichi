import argparse

from tests import test_utils


def test_regression():
    from taichi.examples.autodiff.regression import initialize, regress_raw
    initialize()
    regress_raw()


def pic_regression(result_dir):
    from taichi.examples.autodiff.regression import (initialize, regress_raw,
                                                            number_coeffs, coeffs,
                                                            xs, ys)

    from matplotlib import pyplot as plt
    import numpy as np

    initialize()
    regress_raw()

    curve_xs = np.arange(-2.5, 2.5, 0.01)
    curve_ys = curve_xs * 0
    for i in range(number_coeffs):
        curve_ys += coeffs[i] * np.power(curve_xs, i)

    plt.title(
        'Nonlinear Regression with Gradient Descent (3rd order polynomial)')
    ax = plt.gca()
    ax.scatter(xs, ys, label='data', color='r')
    ax.plot(curve_xs, curve_ys, label='fitted')
    ax.legend()
    ax.grid(True)
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    print(result_dir + '/output.png')

    # Create new directory
    test_utils.mkdir_p(result_dir)
    plt.savefig(result_dir + '/output.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate regression pic')
    parser.add_argument('output_directory',
                        help='output directory of generated pic')
    pic_regression(parser.parse_args().output_directory)
