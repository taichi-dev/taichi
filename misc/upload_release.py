import os
import subprocess
import sys

import requests


def upload_taichi_version():
    username = os.getenv('METADATA_USERNAME')
    password = os.getenv('METADATA_PASSWORD')
    url = os.getenv('METADATA_URL')
    filename = os.listdir('./dist')[0]
    filename = filename[:len(filename) - 4]
    parts = filename.split('-')
    payload = {'version': parts[1], 'platform': parts[4], 'python': parts[2]}
    try:
        response = requests.post(f'https://{url}/add_version/detail',
                                 json=payload,
                                 auth=(username, password),
                                 timeout=5)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as err:
        sys.exit('Updating latest version failed: No internet, ' + str(err))
    except requests.exceptions.HTTPError as err:
        sys.exit('Updating latest version failed: Server error, ' + str(err))
    except requests.exceptions.Timeout as err:
        sys.exit(
            'Updating latest version failed: Time out when connecting server, '
            + str(err))
    except requests.exceptions.RequestException as err:
        sys.exit('Updating latest version failed: ' + str(err))
    response = response.json()
    print(response['message'])


def upload_artifact(is_taichi):
    pwd_env = 'PROD_PWD' if is_taichi else 'NIGHT_PWD'
    twine_password = os.getenv(pwd_env)
    if not twine_password:
        sys.exit(f'Missing password env var {pwd_env}')
    command = ["python3", "-m", "twine", "upload"]
    if not is_taichi:
        command.extend(['--repository', 'testpypi'])
    command.extend(
        ['--verbose', '-u', '__token__', '-p', twine_password, 'dist/*'])
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        sys.exit(f"Twine upload returns error {e.returncode}")


if __name__ == '__main__':
    is_taichi = os.getenv('PROJECT_NAME', 'taichi') == 'taichi'
    upload_artifact(is_taichi)
    if is_taichi:
        upload_taichi_version()
