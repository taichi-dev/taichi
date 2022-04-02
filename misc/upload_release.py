import os
import subprocess
import sys

import requests


def upload_taichi_version():
    username = os.getenv('METADATA_USERNAME')
    password = os.getenv('METADATA_PASSWORD')
    url = os.getenv('METADATA_URL')
    for filename in os.listdir('./dist'):
        filename = filename[:len(filename) - 4]
        parts = filename.split('-')
        payload = {
            'version': parts[1],
            'platform': parts[4],
            'python': parts[2]
        }
        try:
            response = requests.post(f'https://{url}/add_version/detail',
                                     json=payload,
                                     auth=(username, password),
                                     timeout=5)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as err:
            print('Updating latest version failed: No internet,', err)
        except requests.exceptions.HTTPError as err:
            print('Updating latest version failed: Server error,', err)
        except requests.exceptions.Timeout as err:
            print(
                'Updating latest version failed: Time out when connecting server,',
                err)
        except requests.exceptions.RequestException as err:
            print('Updating latest version failed:', err)
        else:
            response = response.json()
            print(response['message'])


def upload_artifact(is_taichi):
    pwd_env = 'PROD_PWD' if is_taichi else 'NIGHT_PWD'
    twine_password = os.getenv(pwd_env)
    if not twine_password:
        sys.exit(f'Missing password env var {pwd_env}')
    command = [sys.executable, '-m', 'twine', 'upload']
    if not is_taichi:
        command.extend(['--repository-url', 'https://pypi.taichi.graphics/simple/'])
    uname = '__token__' if is_taichi else 'admin'
    command.extend(
        ['--verbose', '-u', uname, '-p', twine_password, 'dist/*'])

    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        sys.exit(f"Twine upload returns error {e.returncode}")


if __name__ == '__main__':
    if os.getenv('GITHUB_REPOSITORY',
                 'taichi-dev/taichi') != 'taichi-dev/taichi':
        print('This script should be run from taichi repo')
        sys.exit(0)
    is_taichi = os.getenv('PROJECT_NAME', 'taichi') == 'taichi'
    upload_artifact(is_taichi)
    if is_taichi:
        upload_taichi_version()
