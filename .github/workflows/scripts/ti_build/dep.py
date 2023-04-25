# -*- coding: utf-8 -*-

# -- stdlib --
import shutil
import zipfile
from pathlib import Path
from urllib.parse import urlparse

# -- third party --
import requests

# -- own --
from .misc import get_cache_home, info
from .tinysh import bash, sh, sudo, tar


# -- code --
def unzip(filename, extract_dir, strip=0):
    """
    Unpack zip `filename` to `extract_dir`, optionally stripping `strip` components.
    """
    if not zipfile.is_zipfile(filename):
        raise Exception(f"{filename} is not a zip file")

    extract_dir = Path(extract_dir)

    ar = zipfile.ZipFile(filename)
    try:
        for info in ar.infolist():
            name = info.filename

            # don't extract absolute paths or ones with .. in them
            if name.startswith("/") or ".." in name:
                continue

            target = extract_dir.joinpath(*name.split("/")[strip:]).resolve()
            if not target:
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            if not name.endswith("/"):
                # file
                data = ar.read(info.filename)
                f = open(target, "wb")
                try:
                    f.write(data)
                finally:
                    f.close()
                    del data
    finally:
        ar.close()


def escape_url(url):
    return url.replace("/", "_").replace(":", "_")


SHOULD_USE_MIRROR = {
    "ci": False,
    "aliyun": False,
    "_probed": False,
}


def probe_mirrors():
    if SHOULD_USE_MIRROR["_probed"]:
        return

    try:
        resp = requests.get("http://botmaster.tgr:9000/misc/canary.txt", timeout=0.5)
        if resp.ok and resp.text.strip() == "in-taichi-ci-environment":
            info("Enabling Taichi CI cluster mirror")
            SHOULD_USE_MIRROR["ci"] = True
    except Exception:
        pass

    google_ok = True
    try:
        resp = requests.head("https://google.com", timeout=2)
        google_ok = resp.ok
    except Exception:
        google_ok = False

    if not google_ok:
        info("Enabling Aliyun mirror")
        SHOULD_USE_MIRROR["aliyun"] = True

    SHOULD_USE_MIRROR["_probed"] = True


def download_dep(url, outdir, *, strip=0, force=False, args=None, plain=False, elevate=False):
    """
    Download a dependency archive from `url` and expand it to `outdir`,
    optionally stripping `strip` components.
    """
    outdir = Path(outdir)
    if outdir.exists() and len(list(outdir.glob("*"))) > 0 and not force:
        return

    shutil.rmtree(outdir, ignore_errors=True)

    parsed = urlparse(url)
    name = Path(parsed.path).name
    escaped = escape_url(url)
    depcache = get_cache_home() / "deps"
    depcache.mkdir(parents=True, exist_ok=True)
    local_cached = depcache / escaped

    probe_mirrors()

    urls = [url]

    if SHOULD_USE_MIRROR["aliyun"]:
        urls.append(f"https://taichi-bots.oss-cn-beijing.aliyuncs.com/depcache/{escaped}/{name}")

    if SHOULD_USE_MIRROR["ci"]:
        urls.append(f"http://botmaster.tgr:9000/misc/depcache/{escaped}/{name}")

    size = -1
    for u in reversed(urls):
        try:
            resp = requests.head(u, headers={"Accept-Encoding": "identity"}, timeout=1)
            if resp.ok:
                url = u
                size = int(resp.headers["Content-Length"])
                break
        except Exception:
            pass

    if not local_cached.exists() or local_cached.stat().st_size != size:
        import tqdm

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            prog = tqdm.tqdm(
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                total=total_size,
                desc=name,
            )
            with prog, open(str(local_cached) + ".download", "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    sz = f.write(chunk)
                    prog.update(sz)

        shutil.move(str(local_cached) + ".download", local_cached)

    if name.endswith(".zip"):
        outdir.mkdir(parents=True, exist_ok=True)
        unzip(local_cached, outdir, strip=strip)
    elif name.endswith(".tar.gz") or name.endswith(".tgz"):
        outdir.mkdir(parents=True, exist_ok=True)
        tar("-xzf", local_cached, "-C", outdir, f"--strip-components={strip}")
    elif name.endswith(".sh"):
        bash(local_cached, *args)
    elif "." not in name and args is not None:
        local_cached.chmod(0o755)
        sh.bake(local_cached)(*args)
    elif name.endswith(".exe") and args is not None:
        local_cached.chmod(0o755)
        cmd = sh.bake(local_cached)
        if elevate:
            with sudo():
                cmd(*args)
        else:
            cmd(*args)
        # start(local_cached, *args)
    elif plain:
        outdir.mkdir(parents=True, exist_ok=True)
        shutil.copy(local_cached, outdir / name)
    else:
        raise RuntimeError(f"Unknown file type: {name}")
