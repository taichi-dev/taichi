from git import Repo
import os
repo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
repo = Repo(repo_dir)
commit_hash = str(repo.head.commit)
print(f"Building commit {commit_hash}")

output_fn = os.path.join(repo_dir, 'taichi/common/commit_hash.h')
content = f"#define TI_COMMIT_HASH \"{commit_hash}\"\n"

# First read the file to see if an update is needed
# This reduces unnecessary file changes/linkings
if os.path.exists(output_fn):
    with open(output_fn, 'r') as f:
        old_content = f.read()
        if old_content == content:
            # No update needed
            exit(0)

with open(output_fn, 'w') as f:
    f.write(content)
