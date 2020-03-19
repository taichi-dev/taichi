import sys
import os

msg = os.environ["APPVEYOR_REPO_COMMIT_MESSAGE"]
if msg.startswith('[release]') or sys.version_info[1] == 6:
    exit(
        0
    )  # Build for this configuration (starts with '[release]', or python version is 3.6)
else:
    print(
        f'[appveyor_filer] Not build for [{msg}] with Python {sys.version[:5]}'
    )
    exit(1)  # Do not build this configuration. See appveyor.yml
