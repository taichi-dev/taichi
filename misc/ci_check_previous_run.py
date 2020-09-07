import argparse
import json
import logging
import time
import urllib.request as ur
import sys

API_PREFIX = 'https://api.github.com/repos/taichi-dev/taichi'
SHA = 'sha'
OAUTH_TOKEN = None


def make_api_url(p):
    return f'{API_PREFIX}/{p}'


def send_request(url):
    # https://docs.github.com/en/actions/configuring-and-managing-workflows/authenticating-with-the-github_token#example-calling-the-rest-api
    hdrs = {'Authorization': f'Bearer {OAUTH_TOKEN}'}
    # https://stackoverflow.com/a/47029281/12003165
    req = ur.Request(url, headers=hdrs)
    res = ur.urlopen(req)
    # Headers are defined in https://developer.github.com/v3/rate_limit/
    # https://docs.github.com/en/actions/getting-started-with-github-actions/about-github-actions#usage-limits
    rl = res.getheader('X-Ratelimit-Limit', None)
    rl_remain = res.getheader('X-Ratelimit-Remaining', None)
    logging.debug(
        f'request={url} rate_limit={rl} rate_limit_remaining={rl_remain}')
    return res


def get_commits(pr):
    url = make_api_url(f'pulls/{pr}/commits')
    f = send_request(url)
    return json.loads(f.read())


def locate_previous_commit_sha(commits, head_sha):
    """
    Returns the previous commit of |head_sha| in PR's |commits|.
    """
    assert commits[-1][SHA] == head_sha
    if len(commits) < 2:
        return None
    return commits[-2][SHA]


def get_workflow_runs(page_id):
    # https://docs.github.com/en/rest/reference/actions#list-workflow-runs-for-a-repository
    url = make_api_url(f'actions/runs?page={page_id}')
    f = send_request(url)
    return json.loads(f.read())


def is_desired_workflow(run_json):
    """
    Checks if this run is for the "Presubmit Checks" workflow.
    """
    # Each workflow has a fixed ID.
    # For the "Persubmit Checks" workflow, it is:
    # https://api.github.com/repos/taichi-dev/taichi/actions/workflows/1291024
    DESIRED_ID = 1291024
    return run_json['workflow_id'] == DESIRED_ID


def locate_workflow_run_id(sha):
    done = False
    page_id = 0
    while not done:
        # Note that the REST API to get runs paginates the results.
        runs = get_workflow_runs(page_id)['workflow_runs']
        for r in runs:
            if r['head_sha'] == sha and is_desired_workflow(r):
                return r['id']
        page_id += 1
    return ''


def get_status_of_run(run_id):
    """
    Waits for run identified by |run_id| to complete and returns its status.
    """
    url = make_api_url(f'actions/runs/{run_id}')
    start = time.time()
    retries = 0
    MAX_TIMEOUT = 60 * 60  # 1 hour
    while True:
        f = send_request(url)
        j = json.loads(f.read())
        # https://developer.github.com/v3/checks/runs/#create-a-check-run
        if j['status'] == 'completed':
            c = j['conclusion']
            logging.debug(f'run={run_id} conclusion={c}')
            return c == 'success'

        if time.time() - start > MAX_TIMEOUT:
            return False
        retries += 1
        logging.info(
            f'Waiting to get the status of run={run_id} (url={url}). retries={retries}'
        )
        time.sleep(60)
    return False


def get_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pr', help='PR number')
    parser.add_argument('--sha', help='Head commit SHA in the PR')
    parser.add_argument('--token', help='OAuth token')
    return parser.parse_args()


def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')
    args = get_cmd_args()
    global OAUTH_TOKEN
    OAUTH_TOKEN = args.token

    pr = args.pr
    commits = get_commits(pr)
    num_commits = len(commits)
    logging.debug(f'\nPR={pr} #commits={num_commits}')

    head_sha = args.sha
    prev_sha = locate_previous_commit_sha(commits, head_sha)
    logging.debug(f'SHA: head={head_sha} prev={prev_sha}')
    if prev_sha is None:
        logging.info(f'First commit in PR={pr}, no previous run to check')
        # First commit in the PR
        return 0

    run_id = locate_workflow_run_id(prev_sha)
    if not run_id:
        logging.warning(f'Could not find the workflow run for SHA={prev_sha}')
        return 0

    logging.info(f'Prev commit: SHA={prev_sha} workflow_run_id={run_id}')
    run_ok = get_status_of_run(run_id)
    logging.info(f'workflow_run_id={run_id} ok={run_ok}')
    return 0 if run_ok else 1


if __name__ == '__main__':
    sys.exit(main())
