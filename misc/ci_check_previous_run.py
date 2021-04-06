import argparse
import json
import logging
import sys
import time
import urllib.request as ur

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


CHECK_STILL_RUNNING = 1
CHECK_FOUND_FAILED_JOB = 2
CHECK_ALL_JOBS_SUCCEEDED = 3


def check_all_jobs(jobs):
    # Denotes if there is still any job that has not completed yet.
    still_running = False
    for j in jobs:
        name = j['name']
        if not name.startswith('Build and Test'):
            continue

        job_id = j['id']
        # https://developer.github.com/v3/checks/runs/#create-a-check-run
        status = j['status']
        if status != 'completed':
            logging.debug(
                f'  job={job_id} name={name} still running, status={status}')
            still_running = True
            continue

        concl = j['conclusion']
        if concl != 'success':
            # If we ever find a failed job, the entire check has failed.
            logging.warning(
                f'  job={job_id} name={name} failed, conclusion={concl}')
            return CHECK_FOUND_FAILED_JOB

    return CHECK_STILL_RUNNING if still_running else CHECK_ALL_JOBS_SUCCEEDED


def get_status_of_run(run_id):
    """
    Waits for run identified by |run_id| to complete and returns its status.

    Instead of waiting for the result of the entire workflow run, we only wait
    on those "Build and Test" jobs. The reason is that when jobs like code
    format failed, the entire workflow run will be marked as failed, yet
    @taichi-gardener will automatically make another commit to format the code.
    However, if this check relies on the previous workflow's result, it will be
    marked as failed again...
    """
    url = make_api_url(f'actions/runs/{run_id}/jobs')
    start = time.time()
    retries = 0
    MAX_TIMEOUT = 60 * 60  # 1 hour
    while True:
        f = send_request(url)
        j = json.loads(f.read())
        check_result = check_all_jobs(j['jobs'])
        if check_result != CHECK_STILL_RUNNING:
            ok = check_result == CHECK_ALL_JOBS_SUCCEEDED
            logging.info(f'Done checking the jobs in run={run_id}. ok={ok}')
            return ok

        if time.time() - start > MAX_TIMEOUT:
            logging.warning(f'Timed out waiting for run={run_id}')
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
