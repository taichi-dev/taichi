import argparse
import os
import re
import subprocess

parser = argparse.ArgumentParser(description='Generate all videos of examples')
parser.add_argument('output_directory',
                    help='output directory of generated videos')
output_dir = parser.parse_args().output_directory

example_root = os.path.join('..', 'tests', 'python', 'examples')
for example_dir in os.listdir(example_root):
    full_dir = os.path.join(example_root, example_dir)
    if not os.path.isdir(full_dir):
        continue
    for filename in os.listdir(full_dir):
        match = re.match(r'test_(\w+)\.py', filename)
        if match:
            subprocess.run([
                "python",
                os.path.join(full_dir, filename),
                os.path.join(output_dir, match.group(1))
            ],
                           check=True)
