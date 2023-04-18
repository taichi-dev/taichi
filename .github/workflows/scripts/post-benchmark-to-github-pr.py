#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sqlite3

# -- stdlib --
from pathlib import Path

# -- third party --
import requests

# -- own --

# -- code --
gh = requests.Session()
gh.headers.update(
    {
        "Authorization": f'Bearer {os.environ["GITHUB_TOKEN"]}',
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
)


def post_comment(repo, number, msg):
    gh.post(
        f"https://api.github.com/repos/{repo}/issues/{number}/comments",
        json={"body": msg},
    )


def get_db():
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE release (name TEXT, value REAL)")
    db.execute("CREATE UNIQUE INDEX release_name ON release (name)")
    db.execute("CREATE TABLE current (name TEXT, value REAL)")
    db.execute("CREATE UNIQUE INDEX current_name ON current (name)")
    return db


IGNORE_TAGS = {"type", "release", "impl"}


def flatten_metric(m):
    tags = [f"{k}={v}" for k, v in m["tags"].items() if k not in IGNORE_TAGS]
    tags = ",".join(sorted(tags))
    return (f'{m["name"]}@{tags}', m["value"])


def fmt(v):
    if 0 < abs(v) < 1:
        return f"{v:.3g}"
    else:
        return f"{v:.3f}"


def render_result(baseline, sha, rs):
    texts = []
    _ = texts.append
    _(f"## Benchmark Report")
    _(f"Baseline: `{baseline}`")
    _(f"Current: `{sha}`")
    _("")
    _("| Item | Baseline | Current | Change |")
    _("| --- | --- | --- | --- |")
    for name, cv, rv, rate in rs:
        cmap = ["red", "green"]
        if ":wall_time@" in name:
            cv *= 1000
            rv *= 1000
            cmap = ["green", "red"]

        if rate > 5.0:
            color = cmap[1]
        elif rate < -5.0:
            color = cmap[0]
        else:
            color = "gray"

        rate = f'{" +"[rate > 0]}{rate:.2f}'

        _(rf"| {name} | {fmt(rv)} | {fmt(cv)} | $\textcolor{{{color}}}{{\textsf{{{rate}\\%}}}}$ |")

    return "\n".join(texts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("event")
    parser.add_argument("result")
    options = parser.parse_args()

    db = get_db()

    current = json.loads(Path(options.result).read_text())
    for item in current:
        db.execute("INSERT OR IGNORE INTO current VALUES (?, ?)", flatten_metric(item))

    ver = requests.get("https://benchmark.taichi-lang.cn/releases?order=vnum.desc&limit=1").json()[0]["version"]
    release = requests.get(
        f"https://benchmark.taichi-lang.cn/taichi_benchmark?tags->>type=eq.release&tags->>release=eq.{ver}"
    ).json()

    for item in release:
        db.execute("INSERT OR IGNORE INTO release VALUES (?, ?)", flatten_metric(item))

    rs = db.execute(
        """
        SELECT
            c.name AS name,
            c.value AS cv,
            COALESCE(r.value, 0.0) AS rv,
            COALESCE((c.value - r.value) / r.value * 100, 0.0) AS rate
        FROM
            current c
            LEFT JOIN release r ON (r.name = c.name)
        ORDER BY name
    """
    )

    event = json.loads(Path(options.event).read_text())
    sha = event["client_payload"]["pull_request"]["head"]["sha"]
    text = render_result(ver, sha, rs)

    post_comment(
        event["repository"]["full_name"],
        event["client_payload"]["pull_request"]["number"],
        text,
    )


if __name__ == "__main__":
    main()
