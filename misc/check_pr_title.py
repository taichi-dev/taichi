import sys, os, json, semver, git


def get_old_ver():
    repo = git.Repo('.')
    for c in repo.iter_commits('master', max_count=200):
        if c.summary.startswith('[release]'):
            ver = c.summary.split(']', maxsplit=1)[1]
            if ver[0] == 'v':
                ver = ver[1:]
            ver = ver.split(' ')[0]
            oldver = semver.VersionInfo.parse(ver)
            return oldver
    raise ValueError('Could not find an old version!')


title = sys.argv[1]
print(f'Checking PR title: {title}')

this_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(this_dir, 'prtags.json')
with open(json_path) as f:
    prtags = json.load(f)

if not title.startswith('['):
    exit(f'PR title does not start with any tag: {title}')

if title.endswith(' '):
    exit(f'PR title should not end with a space: {title}')

if title.endswith(']'):
    exit(f'PR title should have bodies regardless tags: {title}')

if '`' in title:
    exit(f'PR title should not contain backquotes (`): {title}')

for x in title.split(']')[1:]:
    if x[0] != ' ':
        exit(f'No space before: {x}')
    if x[1] == ' ':
        exit(f'Extra space before: {x[2:]}')

had_upper = False
for x in title.split('] ')[:-1]:
    if x[0] != '[':
        exit(f'No starting [ for tag: {x}]')
    if x[1:].lower() not in prtags.keys():
        exit(f'Unrecognized PR tag: [{x[1:]}]')
    # 'Misc'.islower() -> False, 'Misc'.isupper() -> False
    # 'misc'.islower() -> True, 'misc'.isupper() -> False
    if not x[1:].islower():
        if had_upper:
            exit(f'At most 1 uppercase tag expected, got: [{x[1:]}]')
        had_upper = True

is_release = False
for x in title.split('] ')[:-1]:
    x = x[1:]
    if x.lower() == 'release':
        is_release = True
        if not x.islower():
            exit(f'[release] must be lowercase, got: [{x}]')

if is_release:
    ts = title.split(']')
    if len(ts) != 2:
        exit(f'Release PRs must have only one tag "[release]", got: {title}')
    ver = ts[1][1:]
    if ver[0] != 'v':
        exit(f'Release version must start with "v", got: {ver}')
    ver = ver[1:]
    try:
        ver = semver.VersionInfo.parse(ver)
    except ValueError:
        exit(f'Invalid SemVer version: {ver}')
    try:
        oldver = get_old_ver()
    except git.exc.GitCommandError:
        pass
    else:
        if ver not in [oldver.bump_minor(), oldver.bump_patch()]:
            exit(f'Version bump incorrect: {oldver} -> {ver}')
else:
    x = title.split(']')[-1].strip()
    if x[0].islower():
        exit(f'PR titles must start with uppercase letters: {x}')

print('OK!')
