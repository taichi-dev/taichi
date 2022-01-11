import os
import re
import subprocess
import sys

if len(sys.argv) != 2:
    print('usage: generate_example_videos.py <output_directory>')
    exit(0)

example_root = os.path.join('..', 'tests', 'python', 'examples')
for example_dir in os.listdir(example_root):
    full_dir = os.path.join(example_root, example_dir)
    for filename in os.listdir(full_dir):
        match = re.match(r'test_(\w+)\.py', filename)
        if match:
            subprocess.run([
                "python",
                os.path.join(full_dir, filename),
                os.path.join(sys.argv[1], match.group(1))
            ])
