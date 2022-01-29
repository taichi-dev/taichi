import json
import os
import platform
import threading
from urllib import request


def check_version():
    # Check Taichi version for the user.
    major = _ti_core.get_version_major()
    minor = _ti_core.get_version_minor()
    patch = _ti_core.get_version_patch()
    version = f'{major}.{minor}.{patch}'
    payload = {'version': version, 'platform': '', 'python': ''}

    system = platform.system()
    if system == 'Linux':
        payload['platform'] = 'manylinux1_x86_64'
    elif system == 'Windows':
        payload['platform'] = 'win_amd64'
    elif system == 'Darwin':
        if platform.release() < '19.0.0':
            payload['platform'] = 'macosx_10_14_x86_64'
        elif platform.machine() == 'x86_64':
            payload['platform'] = 'macosx_10_15_x86_64'
        else:
            payload['platform'] = 'macosx_11_0_arm64'

    python_version = platform.python_version()
    if python_version.startswith('3.6.'):
        payload['python'] = 'cp36'
    elif python_version.startswith('3.7.'):
        payload['python'] = 'cp37'
    elif python_version.startswith('3.8.'):
        payload['python'] = 'cp38'
    elif python_version.startswith('3.9.'):
        payload['python'] = 'cp39'

    # We do not want request exceptions break users' usage of Taichi.
    try:
        payload = json.dumps(payload)
        payload = payload.encode()
        req = request.Request('https://metadata.taichi.graphics/check_version',
                              method='POST')
        req.add_header('Content-Type', 'application/json')
        with request.urlopen(req, data=payload, timeout=5) as response:
            response = json.loads(response.read().decode('utf-8'))
            return response
    except:
        return None


def try_check_version():
    try:
        os.makedirs(_ti_core.get_repo_dir(), exist_ok=True)
        timestamp_path = os.path.join(_ti_core.get_repo_dir(), 'timestamp')
        cur_date = datetime.date.today()
        if os.path.exists(timestamp_path):
            last_time = ''
            with open(timestamp_path, 'r') as f:
                last_time = f.readlines()[0].rstrip()
            if cur_date.strftime('%Y-%m-%d') > last_time:
                response = check_version()
                if response is None:
                    return
                with open(timestamp_path, 'w') as f:
                    f.write((cur_date +
                             datetime.timedelta(days=7)).strftime('%Y-%m-%d'))
                    f.write('\n')
                    if response['status'] == 1:
                        f.write(response['latest_version'])
                    else:
                        f.write('0.0.0')
        else:
            response = check_version()
            if response is None:
                return
            with open(timestamp_path, 'w') as f:
                f.write((cur_date +
                         datetime.timedelta(days=7)).strftime('%Y-%m-%d'))
                f.write('\n')
                if response['status'] == 1:
                    f.write(response['latest_version'])
                else:
                    f.write('0.0.0')
    # Wildcard exception to catch potential file writing errors.
    except:
        pass


def start_version_check_thread():
    skip = os.environ.get("TI_SKIP_VERSION_CHECK")
    if skip != 'ON':
        # We don't join this thread because we do not wish to block users.
        check_version_thread = threading.Thread(target=try_check_version,
                                                daemon=True)
        check_version_thread.start()


__all__ = []
