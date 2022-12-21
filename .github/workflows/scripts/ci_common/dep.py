# -*- coding: utf-8 -*-

# -- stdlib --
import shutil
import zipfile
from pathlib import Path
from urllib.parse import urlparse

# -- third party --
import requests

# -- own --
from .misc import get_cache_home
from .tinysh import tar


# -- code --
def unzip(filename, extract_dir, strip=0):
    '''
    Unpack zip `filename` to `extract_dir`, optionally stripping `strip` components.
    '''
    if not zipfile.is_zipfile(filename):
        raise Exception(f"{filename} is not a zip file")

    extract_dir = Path(extract_dir)

    ar = zipfile.ZipFile(filename)
    try:
        for info in ar.infolist():
            name = info.filename

            # don't extract absolute paths or ones with .. in them
            if name.startswith('/') or '..' in name:
                continue

            target = extract_dir.joinpath(*name.split('/')[strip:]).resolve()
            if not target:
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            if not name.endswith('/'):
                # file
                data = ar.read(info.filename)
                f = open(target, 'wb')
                try:
                    f.write(data)
                finally:
                    f.close()
                    del data
    finally:
        ar.close()


def download_dep(url, outdir, *, strip=0, force=False):
    '''
    Download a dependency archive from `url` and expand it to `outdir`,
    optionally stripping `strip` components.
    '''
    outdir = Path(outdir)
    if outdir.exists() and len(list(outdir.glob('*'))) > 0 and not force:
        return

    shutil.rmtree(outdir, ignore_errors=True)

    parsed = urlparse(url)
    name = Path(parsed.path).name
    escaped = url.replace('/', '_').replace(':', '_')
    depcache = get_cache_home() / 'deps'
    depcache.mkdir(parents=True, exist_ok=True)
    local_cached = depcache / escaped

    if not local_cached.exists():
        cached_url = f'http://botmaster.tgr:9000/misc/depcache/{escaped}/{name}'
        try:
            resp = requests.head(cached_url, timeout=1)
            if resp.ok:
                print('Using near cache: ', cached_url)
                url = cached_url
        except Exception:
            pass

        import tqdm

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            prog = tqdm.tqdm(unit="B",
                             unit_scale=True,
                             unit_divisor=1024,
                             total=total_size,
                             desc=name)
            with prog, open(local_cached, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    sz = f.write(chunk)
                    prog.update(sz)

    outdir.mkdir(parents=True, exist_ok=True)

    if name.endswith('.zip'):
        unzip(local_cached, outdir, strip=strip)
    elif name.endswith('.tar.gz') or name.endswith('.tgz'):
        tar('-xzf', local_cached, '-C', outdir, f'--strip-components={strip}')
    else:
        raise RuntimeError(f'Unknown file type: {name}')
