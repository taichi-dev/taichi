#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import subprocess

import requests

gh = requests.Session()
gh.headers.update({
    'Authorization': f'Bearer {os.environ["GITHUB_TOKEN"]}',
    'Accept': 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2022-11-28',
})
ev = json.loads(open(os.environ['GITHUB_EVENT_PATH'], 'r').read())


def must(cond, msg):
    if not cond:
        print(msg)
        gh.post(
            f'https://api.github.com/repos/{ev["repository"]}/issues/{ev["event"]["number"]}/comments',
            json={
                'body': f'ghstack bot failed: {msg}',
            })
        exit(1)


def main():
    must(ev['event_name'] == 'pull_request', 'Not a pull request')
    head_ref = ev['head_ref']
    must(head_ref and re.match(r'^gh/[A-Za-z0-9-]+/[0-9]+/head$', head_ref),
         'Not a ghstack PR')
    orig_ref = head_ref.replace('/head', '/orig')
    print(':: Fetching newest master...')
    must(os.system('git fetch origin master') == 0, "Can't fetch master")
    print(':: Fetching orig branch...')
    must(
        os.system(f'git fetch origin {orig_ref}') == 0,
        "Can't fetch orig branch")

    proc = subprocess.Popen(
        'git log FETCH_HEAD...$(git merge-base FETCH_HEAD origin/master)',
        stdout=subprocess.PIPE,
        shell=True)
    out, _ = proc.communicate()
    must(proc.wait() == 0, '`git log` command failed!')

    prs = re.findall(
        r'Pull Request resolved: https://github.com/.*?/pull/([0-9]+)',
        out.decode('utf-8'))
    prs = list(map(int, prs))
    must(prs and prs[0] == ev['event']['number'],
         'Extracted PR numbers not seems right!')

    for pr in prs:
        print(f':: Checking PR status #{pr}... ', end='')
        resp = gh.get(
            f'https://api.github.com/repos/{ev["repository"]}/pulls/{pr}')
        must(resp.ok, 'Error Getting PR Object!')
        pr_obj = resp.json()
        resp = gh.get(
            f'https://api.github.com/repos/{ev["repository"]}/commits/{pr_obj["head"]["sha"]}/check-runs'
        )
        must(resp.ok, 'Error Getting Check Runs Status!')
        checkruns = resp.json()
        for cr in checkruns['check_runs']:
            status = cr.get('conclusion', cr['status'])
            name = cr['name']
            must(
                status == 'success',
                f'PR #{pr} check-run `{name}`\'s status `{status}` is not success!'
            )
        print('SUCCESS!')

    print(':: All PRs are ready to be landed!')


if __name__ == '__main__':
    main()
