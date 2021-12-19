import os
from datetime import date

import requests

version = os.getenv('RELEASE_VERSION')
version = version[1:]
version_num = version.split('.')
major = int(version_num[0])
minor = int(version_num[1])
patch = int(version_num[2])
release_date = date.today().strftime('%Y-%m-%d')

payload = {
    'version': version,
    'major': major,
    'minor': minor,
    'patch': patch,
    'date': release_date
}

username = os.getenv('METADATA_USERNAME')
password = os.getenv('METADATA_PASSWORD')
url = os.getenv('METADATA_URL')

try:
    response = requests.post(f'https://{url}/add_version/main',
                             json=payload,
                             auth=(username, password),
                             timeout=5)
    response.raise_for_status()
except requests.exceptions.ConnectionError as err:
    print('Updating latest version failed: No internet,', err)
    exit(1)
except requests.exceptions.HTTPError as err:
    print('Updating latest version failed: Server error,', err)
    exit(1)
except requests.exceptions.Timeout as err:
    print('Updating latest version failed: Time out when connecting server,',
          err)
    exit(1)
except requests.exceptions.RequestException as err:
    print('Updating latest version failed:', err)
    exit(1)

response = response.json()
print(response['message'])
