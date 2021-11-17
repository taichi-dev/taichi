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

response = requests.post('http://54.90.48.192/add_version/main',
                         json=payload,
                         auth=requests.auth.HTTPBasicAuth(username, password))
print(response.text)
