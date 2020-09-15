import argparse
import functools
import logging
import re
import subprocess
from functools import partial, reduce
from pathlib import Path
from typing import Callable, List, Match, Tuple, Union

import git
import requests
from requests.auth import HTTPBasicAuth


API_PREFIX = 'https://api.github.com/repos/taichi-dev/taichi'
logger = logging.getLogger(__name__)


def make_api_url(p):
    return f'{API_PREFIX}/{p}'


def generate_changelog() -> str:
    # redirect the stdout to DEVNULL to make it less noisy
    subprocess.run(["ti changelog --save"],
                   shell=True,
                   check=True,
                   stdout=subprocess.DEVNULL)
    # need to make sure call from the root of the taichi repo
    with open("CHANGELOG.md", "r") as fp:
        changelog = fp.read()
    return changelog


def regenerate_docs():
    # redirect both stdout and stderr to DEVNULL to make it less noisy
    subprocess.run(["cmake ."],
                   shell=True,
                   check=True,
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)


def _semver_matcher() -> Callable:
    """Return a func performs regex matching for semantic versions.
       The regex uses word_boundry and is sensitive to the changes
       to the contents of the CMakeLists.txt file. The match should
       have 3 matched groups:
       1. `SET(TI_VERSION_MAJOR `
       2. the corresponding version number
       3. `)`
    """
    return lambda s, w: re.search(rf"(\bSET\(TI_VERSION_{w} )(\d+)(\b\))", s)


def _intify_version(v: Match) -> int:
    """Convert matched group v to integer."""
    return int(v.group(2))


def parse_semver(
    cmakelist_path: str,
    return_match_groups: bool = False
) -> Union[Tuple[int, int, int], Tuple[Match[str], Match[str], Match[str]]]:
    """Parse and return the major, minor and patch version numbers
       (or matched groups) from CMakeLists.txt given CMAKELIST_PATH.
    """
    with open(cmakelist_path, "r") as fp:
        cmakelist = fp.read()
    matcher = partial(_semver_matcher(), cmakelist)
    major, minor, patch = map(matcher, ["MAJOR", "MINOR", "PATCH"])
    if return_match_groups:
        return major, minor, patch
    return tuple(map(_intify_version, (major, minor, patch)))


def bump_major(cmakelist_path: str) -> Tuple[int, int, int]:
    """Semantically bump the major version in CMakeLists.txt given CMAKELIST_PATH
       in-place as side-effect, return the result semVer."""
    major, minor, patch = parse_semver(cmakelist_path=cmakelist_path,
                                       return_match_groups=True)
    patterns = (
        (major.re, rf"\g<1>{_intify_version(major) + 1}\g<3>"),
        (minor.re, rf"\g<1>0\g<3>"),
        (patch.re, rf"\g<1>0\g<3>"),
    )
    new_cmakelist_content = reduce(
        lambda content, pattern: re.sub(*pattern, content), patterns,
        major.string)
    with open(cmakelist_path, "w") as fp:
        fp.write(new_cmakelist_content)
    return _intify_version(major) + 1, 0, 0


def bump_minor(cmakelist_path: str) -> Tuple[int, int, int]:
    """Semantically bump the minor version in CMakeLists.txt given CMAKELIST_PATH
       in-place as side-effect, return the result semVer."""
    major, minor, patch = parse_semver(cmakelist_path=cmakelist_path,
                                       return_match_groups=True)
    patterns = (
        (minor.re, rf"\g<1>{_intify_version(minor) + 1}\g<3>"),
        (patch.re, rf"\g<1>0\g<3>"),
    )
    new_cmakelist_content = reduce(
        lambda content, pattern: re.sub(*pattern, content), patterns,
        minor.string)
    with open(cmakelist_path, "w") as fp:
        fp.write(new_cmakelist_content)
    return _intify_version(major), _intify_version(minor) + 1, 0


def bump_patch(cmakelist_path: str) -> Tuple[int, int, int]:
    """Semantically bump the patch version in CMakeLists.txt given CMAKELIST_PATH
       in-place as side-effect, return the result semVer."""
    major, minor, patch = parse_semver(cmakelist_path=cmakelist_path,
                                       return_match_groups=True)
    new_cmakelist_content = re.sub(patch.re,
                                   rf"\g<1>{_intify_version(patch) + 1}\g<3>",
                                   patch.string)
    with open(cmakelist_path, "w") as fp:
        fp.write(new_cmakelist_content)
    return (*map(_intify_version, [major, minor]), _intify_version(patch) + 1)


def create_branch_and_push_ref(repo: git.Repo, feature_branch: str):
    """Create a new FEATURE_BRANCH in REPO and push the ref to its remote origin."""
    origin = repo.remotes.origin
    repo.create_head(feature_branch).checkout()
    repo.git.push("--set-upstream", origin, repo.head.ref)


def commit_files_and_push(repo: git.Repo, message: str, author: git.Actor,
                          committer: git.Actor, files: List[str]):
    """Add local FILES to git, commit them to REPO's remote origin with MESSAGE,
       AUTHOR and COMMITTER info."""
    origin = repo.remotes.origin
    repo.index.add(files)
    repo.index.commit(message=message, author=author, committer=committer)
    origin.push()


def make_pull_request(auth_info: HTTPBasicAuth,
                      title: str,
                      feature_branch: str,
                      content: str,
                      base_branch: str = "master") -> int:
    """Make a pull request from FEATURE_BRANCH to BASE_BRANCH with TITLE and CONTENT,
       AuthN is done via AUTH_INFO. Return the number of the pull request or
       throw HTTPError. This is not idempotent."""
    url = make_api_url("pulls")
    headers = {"Accept": "application/vnd.github.v3+json"}
    payload = {
        "title": title,
        "head": feature_branch,
        "base": base_branch,
        "body": content,
        "maintainer_can_modify": True,
        "draft": False,
    }
    response = requests.post(url=url,
                             headers=headers,
                             json=payload,
                             auth=auth_info)
    response.raise_for_status()
    return response.json()["number"]


def merge_pull_request(auth_info: HTTPBasicAuth, pull_number: int) -> str:
    """Squash and merge a pull request identified by PULL_NUMBER, AuthN is done
       via AUTH_INFO. Return the merge commit SHA hash or throw HTTPError.
       This is not idempotent."""
    url = make_api_url(f"pulls/{pull_number}/merge")
    headers = {"Accept": "application/vnd.github.v3+json"}
    payload = {
        "merge_method": "squash",
    }
    response = requests.put(url=url,
                            headers=headers,
                            json=payload,
                            auth=auth_info)
    response.raise_for_status()
    return response.json()["sha"]


def update_branch_ref(auth_info: HTTPBasicAuth,
                      commit_hash: str,
                      target_branch: str = "stable",
                      force: bool = True):
    """Update the ref of TARGET_BRANCH to point to COMMIT_HASH,
       AuthN is done via AUTH_INFO. This is idempotent."""
    url = make_api_url(f"git/refs/heads/{target_branch}")
    headers = {"Accept": "application/vnd.github.v3+json"}
    payload = {"sha": commit_hash, "force": force}
    response = requests.patch(url=url,
                              headers=headers,
                              json=payload,
                              auth=auth_info)
    response.raise_for_status()


def create_release(auth_info: HTTPBasicAuth, tag_name: str, name: str,
                   commit_hash: str, content: str):
    """Create a new release with NAME and CONTENT points at COMMIT_HASH and
       tagged as TAG_NAME, AuthN is done via AUTH_INFO. This is not idempotent."""
    url = make_api_url("releases")
    headers = {"Accept": "application/vnd.github.v3+json"}
    payload = {
        "tag_name": tag_name,
        # using commit hash than "master" here prevents race conditions from happening
        "target_commitish": commit_hash,
        "name": name,
        "body": content,
        "draft": False,
        "prerelease": False,
    }
    response = requests.post(url=url,
                             headers=headers,
                             json=payload,
                             auth=auth_info)
    response.raise_for_status()


def get_cmd_args():
    parser = argparse.ArgumentParser(
        description="Taichi release automation cli")
    parser.add_argument("release_type",
                        choices=["major", "minor", "patch"],
                        help="The type of the semantic release")
    parser.add_argument(
        "-t",
        "--token",
        type=str,
        required=True,
        help="OAuth token to send authenticated requests to Github API")
    parser.add_argument("-a",
                        "--author",
                        type=str,
                        default="Taichi Gardener",
                        help="The name of the author of the release")
    parser.add_argument("-e",
                        "--email",
                        type=str,
                        default="taichigardener@gmail.com",
                        help="The email of the author of the release")
    return parser.parse_args()


def main(args):
    # 0. Initialize release metadata
    logger.info("=> 0. Initialize release metadata")
    repo = git.Repo(".")
    author = committer = git.Actor(args.author, args.email)

    # TODO: CONVERT TO USE OAUTH TOKEN
    authentication = HTTPBasicAuth("PERSON", "PERSONAL TOKEN")
    # TODO: ALSO NEED TO SETUP GIT CLIENT

    # Parse out the semantic versions
    major_, minor_, patch_ = parse_semver(cmakelist_path="./CMakeLists.txt",
                                          return_match_groups=False)

    # 1. Dispatch on release types, bump the version and regenerate docs
    logger.info(
        f"=> 1. Bump the {args.release_type} version and regenerate docs")
    if args.release_type == "major":
        major, minor, patch = bump_major(cmakelist_path="./CMakeLists.txt")
    elif args.release_type == "minor":
        major, minor, patch = bump_minor(cmakelist_path="./CMakeLists.txt")
    else:
        major, minor, patch = bump_patch(cmakelist_path="./CMakeLists.txt")
    regenerate_docs()
    logger.info(
        f"=>\tThe version is bumped from {major_}.{minor_}.{patch_} to {major}.{minor}.{patch}"
    )

    # 2. Create a branch for the release
    logger.info("=> 2. Create a branch for the release")
    release_branch = f"release-{major}.{minor}.{patch}"
    create_branch_and_push_ref(repo=repo, feature_branch=release_branch)
    logger.info(f"=>\tBranch {release_branch} has been created")

    # 3. Add, commit and push changes
    logger.info("=> 3. Add, commit and push changes")
    files = ["docs", "CMakeLists.txt"]
    commit_files_and_push(repo=repo,
                          message=f"[release] v{major}.{minor}.{patch}",
                          author=author,
                          committer=committer,
                          files=files)

    # 4. Generate the changelog
    logger.info("=> 4. Generate the changelog")
    changelog = generate_changelog()

    # 5. Make the release PR
    logger.info("=> 5. Make the release PR")
    pull_request = make_pull_request(
        title=f"[release] v{major}.{minor}.{patch}",
        feature_branch=release_branch,
        content=changelog,
        auth_info=authentication,
        base_branch="master")
    logger.info(f"=>\tA release PR #{pull_request} has been created")

    # TODO: NEED TO WAIT THE TESTS TO PASS TO MERGE

    # TODO: EITHER WAIT FOR A FEW SECONDS OR KEEP CHECKING MERGEABLITY WITH RETRIES AND TIMEOUTS
    # 6. Merge the release PR
    logger.info("=> 6. Merge the release PR")
    merge_commit = merge_pull_request(auth_info=authentication,
                                      pull_number=pull_request)
    logger.info(
        f"=>\tRelease PR #{pull_request} has been merged as commit {merge_commit}"
    )

    # TODO: NEED TO WAIT THE BUILDBOTS TO PASS BEFORE RELEASE
    # 7. Update stable branch to point to the latest release commit
    logger.info(
        "=> 7. Update stable branch to point to the latest release commit")
    update_branch_ref(auth_info=authentication,
                      commit_hash=merge_commit,
                      target_branch="stable")
    logger.info(f"=>\tThe stable branch has been updated to {merge_commit}")

    # 8. Create the official release
    logger.info("=> 8. Create the official release")
    create_release(auth_info=authentication,
                   tag_name=f"v{major}.{minor}.{patch}",
                   name=f"v{major}.{minor}.{patch}",
                   commit_hash=merge_commit,
                   content=changelog)
    logger.info(
        f"=>\tA Github release v{major}.{minor}.{patch} has been created")


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    arguments = get_cmd_args()
    main(args=arguments)
