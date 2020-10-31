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
    logging.debug('request=%s rate_limit=%s rate_limit_remaining=%s', url, rl,
                  rl_remain)
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
    Checks if this run is for the "Persubmit Checks" workflow.
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


def cancel_workflow_run(run_id):
    """
    Cancels the workflow run identified by |run_id|.

    I gave up trying to utilize the existing actions such as
    https://github.com/marketplace/actions/cancel-workflow-action
    It kept giving me "Error while cancelling workflow_id 1291024:
    Resource not accessible by integration"

    Some issues reported around this:
    * https://github.com/styfle/cancel-workflow-action/issues/7
    * https://github.com/styfle/cancel-workflow-action/issues/8
    * https://github.community/t/github-actions-are-severely-limited-on-prs/18179#M9249
    and many more...

    Whatever, github.
    """
    url = make_api_url(f'actions/runs/{run_id}')
    MAX_RETRIES = 20
    for r in range(MAX_RETRIES):
        f = send_request(url)
        j = json.loads(f.read())

        status = j['status']
        if status not in {'queued', 'in_progress'}:
            logging.info('Wrkflow run=%s done, conclusion=%s', run_id,
                         j['conclusion'])
            return True

        cancel_url = j['cancel_url']
        f = send_request(cancel_url)
        logging.debug(
            '[%d] Issued cancel requeest (url=%s) to workflow run=%s, waiting for the status...',
            r, cancel_url, run_id)
        time.sleep(30)
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
    logging.debug('\nPR=%s #commits=%d', pr, num_commits)

    head_sha = args.sha
    prev_sha = locate_previous_commit_sha(commits, head_sha)
    logging.debug(f'SHA: head=%s prev=%s', head_sha, prev_sha)
    if prev_sha is None:
        logging.info('First commit in PR=%s, no previous run to check', pr)
        # First commit in the PR
        return 0

    run_id = locate_workflow_run_id(prev_sha)
    if not run_id:
        logging.warning('Could not find the workflow run for SHA=%s', prev_sha)
        return 0

    logging.info('Prev commit: SHA=%s workflow_run_id=%s', prev_sha, run_id)
    done = cancel_workflow_run(run_id)
    return 0 if done else 1


if __name__ == '__main__':
    sys.exit(main())
