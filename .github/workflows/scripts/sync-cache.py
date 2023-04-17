#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -- prioritized --
import ti_build  # isort: skip, early initialization happens here

# -- stdlib --
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse

# -- third party --
import requests
import tqdm

# -- own --
from ti_build.bootstrap import ensure_dependencies
from ti_build.dep import escape_url

# -- code --
ensure_dependencies("boto3")
RE = re.compile(r"(https?:\/\/[A-Za-z0-9\-./_%]+\.(tar\.gz|tgz|zip|exe|sh))", re.I)
base = Path(__file__).parent


def walk(path):
    for f in path.iterdir():
        if f.is_dir():
            yield from walk(f)
        else:
            yield f


def find_urls():
    for f in walk(base):
        if f.suffix not in (".py", ".sh", ".ps1", ".yml", ".yaml", ""):
            continue

        with f.open() as f:
            urls = RE.findall(f.read())
            for url in urls:
                yield url[0]


def download(url):
    """
    Download to temp file
    """
    f = tempfile.TemporaryFile()
    parsed = urlparse(url)
    name = Path(parsed.path).name

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        prog = tqdm.tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            total=total_size,
            desc=f"💾 {name}",
        )
        with prog:
            for chunk in r.iter_content(chunk_size=8192):
                sz = f.write(chunk)
                prog.update(sz)

    return f


def upload(cli, prompt, bucket, path, f):
    """
    Upload to cache
    """
    total_size = f.seek(0, 2)
    f.seek(0, 0)
    prog = tqdm.tqdm(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        total=total_size,
        desc=f"📤 {prompt}",
    )

    with prog:
        orig, f.close = f.close, lambda: None
        cli.upload_fileobj(f, bucket, path, Callback=prog.update)
        f.close = orig


def probe(url):
    try:
        resp = requests.head(url, timeout=5)
        if resp.ok:
            return True
    except Exception:
        pass

    return False


def make_cli(endpoint, key_id, key_secret, addr_style="path"):
    import boto3
    from botocore.client import Config

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key_id,
        aws_secret_access_key=key_secret,
        config=Config(s3={"addressing_style": addr_style}),
    )


def main():
    mastercli = make_cli(
        "http://botmaster.tgr:9000",
        os.environ["BOT_MINIO_ACCESS_KEY"],
        os.environ["BOT_MINIO_SECRET_KEY"],
    )
    osscli = make_cli(
        "https://oss-cn-beijing.aliyuncs.com",
        os.environ["BOT_OSS_ACCESS_KEY"],
        os.environ["BOT_OSS_SECRET_KEY"],
        addr_style="virtual",
    )

    for url in find_urls():
        f = None
        print(f"🔍 {url}")
        escaped = escape_url(url)
        name = Path(urlparse(url).path).name

        if not probe(f"http://botmaster.tgr:9000/misc/depcache/{escaped}/{name}"):
            f = f or download(url)
            upload(mastercli, "Near Cache", "misc", f"depcache/{escaped}/{name}", f)

        if not probe(f"https://taichi-bots.oss-cn-beijing.aliyuncs.com/depcache/{escaped}/{name}"):
            f = f or download(url)
            upload(osscli, "Aliyun OSS", "taichi-bots", f"depcache/{escaped}/{name}", f)


if __name__ == "__main__":
    main()
