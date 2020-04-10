from git import Repo
import os
repo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
repo = Repo(repo_dir)
commit_hash = str(repo.head.commit)
print(f"Building commit {commit_hash}")

output_fn = os.path.join(repo_dir, 'taichi/common/commit_hash.h')
with open(output_fn, 'w') as f:
    f.write(f"#define TI_COMMIT_HASH \"{commit_hash}\"\n")
