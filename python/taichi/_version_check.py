import datetime
import json
import os
import platform
import threading
import uuid
from urllib import request

from taichi._lib import core as _ti_core


def check_version(cur_uuid):
    # Check Taichi version for the user.
    major = _ti_core.get_version_major()
    minor = _ti_core.get_version_minor()
    patch = _ti_core.get_version_patch()
    version = f"{major}.{minor}.{patch}"
    payload = {"version": version, "platform": "", "python": ""}

    system = platform.system()
    if system == "Linux":
        payload["platform"] = "manylinux_2_27_x86_64"
    elif system == "Windows":
        payload["platform"] = "win_amd64"
    elif system == "Darwin":
        if platform.release() < "19.0.0":
            payload["platform"] = "macosx_10_14_x86_64"
        elif platform.machine() == "x86_64":
            payload["platform"] = "macosx_10_15_x86_64"
        else:
            payload["platform"] = "macosx_11_0_arm64"

    python_version = platform.python_version().split(".")
    payload["python"] = "cp" + python_version[0] + python_version[1]

    payload["uuid"] = cur_uuid
    if os.getenv("TI_CI") == "1":
        payload["type"] = "CI"
    # We do not want request exceptions break users' usage of Taichi.
    try:
        payload = json.dumps(payload)
        payload = payload.encode()
        req = request.Request("https://metadata.taichi.graphics/check_version", method="POST")
        req.add_header("Content-Type", "application/json")
        with request.urlopen(req, data=payload, timeout=5) as response:
            response = json.loads(response.read().decode("utf-8"))
            return response
    except:
        return None


def write_version_info(response, cur_uuid, version_info_path, cur_date):
    if response is None:
        return
    with open(version_info_path, "w") as f:
        f.write((cur_date).strftime("%Y-%m-%d"))
        f.write("\n")
        if response["status"] == 1:
            f.write(response["latest_version"])
        else:
            f.write("0.0.0")
        f.write("\n")
        f.write(cur_uuid)
        f.write("\n")


def try_check_version():
    try:
        os.makedirs(_ti_core.get_repo_dir(), exist_ok=True)
        version_info_path = os.path.join(_ti_core.get_repo_dir(), "version_info")
        cur_date = datetime.date.today()
        if os.path.exists(version_info_path):
            with open(version_info_path, "r") as f:
                version_info_file = f.readlines()
                last_time = version_info_file[0].rstrip()
                cur_uuid = version_info_file[2].rstrip()
            if cur_date.strftime("%Y-%m-%d") > last_time:
                response = check_version(cur_uuid)
                write_version_info(response, cur_uuid, version_info_path, cur_date)
        else:
            cur_uuid = str(uuid.uuid4())
            write_version_info({"status": 0}, cur_uuid, version_info_path, cur_date)
            response = check_version(cur_uuid)
            write_version_info(response, cur_uuid, version_info_path, cur_date)
    # Wildcard exception to catch potential file writing errors.
    except:
        pass


def start_version_check_thread():
    skip = os.environ.get("TI_SKIP_VERSION_CHECK")
    if skip != "ON":
        # We don't join this thread because we do not wish to block users.
        check_version_thread = threading.Thread(target=try_check_version, daemon=True)
        check_version_thread.start()


__all__ = []
