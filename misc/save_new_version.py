import os
import requests
from datetime import date

version = os.getenv('RELEASE_VERSION')
version = version[1:]
version_num = version.split('.')
major = int(version_num[0])
minor = int(version_num[1])
patch = int(version_num[2])
release_date = date.today().strftime('%Y-%m-%d')

payload = {'version': version, 'major': major, 'minor': minor, 'patch': patch, 'date': release_date}

username = os.getenv('METADATA_USERNAME')
password = os.getenv('METADATA_PASSWORD')
url = os.getenv('METADATA_URL')

response = requests.post('http://'+url+'/add_version/main', json=payload, auth=(username, password))
print(response.text)
