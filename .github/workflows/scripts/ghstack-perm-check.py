#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import subprocess
import sys

import requests


def main():
    gh = requests.Session()
    gh.headers.update(
        {
            "Authorization": f'Bearer {os.environ["GITHUB_TOKEN"]}',
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    )
    EV = json.loads(sys.stdin.read())
    REPO = EV["repository"]
    PR = EV["event"]["client_payload"]["pull_request"]
    NUMBER = PR["number"]

    def must(cond, msg):
        if not cond:
            print(msg)
            gh.post(
                f"https://api.github.com/repos/{REPO}/issues/{NUMBER}/comments",
                json={
                    "body": f"ghstack bot failed: {msg}",
                },
            )
            exit(1)

    head_ref = PR["head"]["ref"]
    must(
        head_ref and re.match(r"^gh/[A-Za-z0-9-]+/[0-9]+/head$", head_ref),
        "Not a ghstack PR",
    )
    orig_ref = head_ref.replace("/head", "/orig")
    print(":: Fetching newest master...")
    must(os.system("git fetch origin master") == 0, "Can't fetch master")
    print(":: Fetching orig branch...")
    must(os.system(f"git fetch origin {orig_ref}") == 0, "Can't fetch orig branch")

    proc = subprocess.Popen(
        "git log FETCH_HEAD...$(git merge-base FETCH_HEAD origin/master)",
        stdout=subprocess.PIPE,
        shell=True,
    )
    out, _ = proc.communicate()
    must(proc.wait() == 0, "`git log` command failed!")

    pr_numbers = re.findall(
        r"Pull Request resolved: https://github.com/.*?/pull/([0-9]+)",
        out.decode("utf-8"),
    )
    pr_numbers = list(map(int, pr_numbers))
    must(pr_numbers and pr_numbers[0] == NUMBER, "Extracted PR numbers not seems right!")

    for n in pr_numbers:
        print(f":: Checking PR status #{n}... ", end="")
        resp = gh.get(f"https://api.github.com/repos/{REPO}/pulls/{n}")
        must(resp.ok, "Error Getting PR Object!")
        pr_obj = resp.json()

        resp = gh.get(f"https://api.github.com/repos/{REPO}/pulls/{NUMBER}/reviews")
        must(resp.ok, "Error Getting PR Reviews!")
        reviews = resp.json()
        idmap = {}
        approved = False
        for r in reviews:
            s = r["state"]
            if s not in ("COMMENTED",):
                idmap[r["user"]["login"]] = r["state"]

        for u, cc in idmap.items():
            approved = approved or cc == "APPROVED"
            must(
                cc in ("APPROVED", "DISMISSED"),
                f"@{u} has stamped PR #{n} `{cc}`, please resolve it first!",
            )

        must(approved, f"PR #{n} is not approved yet!")

        resp = gh.get(f'https://api.github.com/repos/{REPO}/commits/{pr_obj["head"]["sha"]}/check-runs')
        must(resp.ok, "Error getting check runs status!")
        checkruns = resp.json()
        for cr in checkruns["check_runs"]:
            status = cr.get("conclusion", cr["status"])
            name = cr["name"]
            if name == "Copilot for PRs":
                continue
            must(
                status in ("success", "neutral"),
                f"PR #{n} check-run `{name}`'s status `{status}` is not success!",
            )
        print("SUCCESS!")

    print(":: All PRs are ready to be landed!")


if __name__ == "__main__":
    main()
