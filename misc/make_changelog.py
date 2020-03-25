from git import Repo

g = Repo('.')
commits = list(g.iter_commits('master', max_count=200))
begin, end = -1, 0

def format(c):
    return f'   - {c.summary} (by **{c.author}**)'

for i, c in enumerate(commits):
    s = c.summary
    if s.startswith('[release]'):
        if begin == -1:
            begin = i
        else:
            print(format(c))
            break
    if begin != -1:
        print(format(c))
