import argparse
import os

from deserialize import ResultsBuilder

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-f',
                        '--folder',
                        default='./results',
                        dest='folder',
                        type=str,
                        help='Path of result folder. Defaults to ./results')

    parser.add_argument('-H',
                        '--host',
                        default='localhost',
                        dest='host',
                        type=str,
                        help='Provide destination host. Defaults to localhost')

    parser.add_argument('-p',
                        '--port',
                        default='5006',
                        dest='port',
                        type=str,
                        help='Provide destination port. Defaults to 5006')

    args = parser.parse_args()
    benchmarks_root_dir = os.path.dirname(os.path.realpath(__file__))
    visualization_file_path = os.path.join(benchmarks_root_dir,
                                           'visualization')

    results = ResultsBuilder(args.folder)
    results.save_results_as_json(visualization_file_path)
    print(f'path of result folder: {args.folder}')
    print(f'save results.json to: {visualization_file_path}')

    visualization_cmd_str = f'bokeh serve --show {visualization_file_path}'
    visualization_mode = 'Local Mode'
    if args.host != 'localhost':
        visualization_mode = 'Remote Mode'
        visualization_cmd_str = f'bokeh serve {visualization_file_path} --allow-websocket-origin={args.host}:{args.port}'
    print(
        f'[{visualization_mode}] running at: http://{args.host}:{args.port}/visualization)'
    )

    os.system(visualization_cmd_str)
