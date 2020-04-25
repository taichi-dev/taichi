from git import Repo
import os
repo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
repo = Repo(repo_dir)
commit_hash = str(repo.head.commit)
print(f"Building commit {commit_hash}")
exit()

output_fn = os.path.join(repo_dir, 'taichi/common/commit_hash.h')
try:
    with open(output_fn, 'r') as f:
        old_hash = f.read().split('"')[1]  # parse last commit hash
        if commit_hash == old_hash:
            exit()  # not changed
except:
    pass
with open(output_fn, 'w') as f:
    f.write(f"#define TI_COMMIT_HASH \"{commit_hash}\"\n")
