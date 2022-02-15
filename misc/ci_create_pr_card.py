import json
import os
from typing import Any, List, Mapping

from github import Github
from github.Project import Project
from github.Repository import Repository


def load_project_map() -> Mapping[str, str]:
    with open(os.path.join(os.path.dirname(__file__),
                           'tag_to_project.json')) as f:
        return json.load(f)


PROJECT_MAP = load_project_map()


def extract_tags(title: str) -> List[str]:
    """
    Extract tags from PR title like "[ci] [bug] fix a bug"
    """
    tags: List[str] = []
    for x in title.split('] ')[:-1]:
        if x[0] != '[':
            raise ValueError(f'No starting [ for tag: {x}]')
        tags.append(x[1:].lower())
    return tags


def get_project(repo: Repository, name: str) -> Project:
    """
    Get project from repository by name
    """
    for project in repo.get_projects():
        if project.name == name:
            return project
    raise ValueError(f'No project with name: {name}')


def _create_pr_card(pr: dict, project: Project) -> None:
    to_do_column = next(iter(project.get_columns()))
    print(f"Creating card for PR #{pr['number']} in project {project.name}")
    to_do_column.create_card(content_id=pr['id'], content_type="PullRequest")


def _remove_pr_card(pr: dict, project: Project) -> None:
    to_do_column = next(iter(project.get_columns()))
    for card in to_do_column.get_cards():
        if not card.content_url:
            continue
        if card.content_url.split('/')[-1] == str(pr['number']):
            print(f"Deleting PR #{pr['number']} from project {project.name}")
            card.delete()
            return
    print(
        f"PR #{pr['number']} doesn't exist in the To-do column of project {project.name}"
    )


def create_pr_card(event: Mapping[str, Any]) -> None:
    new_projects = {
        PROJECT_MAP[tag]
        for tag in extract_tags(event['pull_request']['title'])
        if tag in PROJECT_MAP
    }
    gh = Github(os.environ['GITHUB_TOKEN'])
    repo = gh.get_repo(event['repository']['full_name'])
    pr = event['pull_request']
    if event['action'] == 'opened':
        for project_name in new_projects:
            _create_pr_card(pr, get_project(repo, project_name))
    else:
        old_title = event.get("changes", {}).get("title", {}).get("from")
        if not old_title:
            print("PR title isn't changed, nothing to do")
            return
        old_projects = {
            PROJECT_MAP[tag]
            for tag in extract_tags(old_title) if tag in PROJECT_MAP
        }
        to_remove = old_projects - new_projects
        to_add = new_projects - old_projects
        for project_name in to_remove:
            _remove_pr_card(pr, get_project(repo, project_name))
        for project_name in to_add:
            _create_pr_card(pr, get_project(repo, project_name))


def main() -> None:
    event = json.loads(os.environ['GH_EVENT'])
    create_pr_card(event)


def test():
    event = {
        "action": "opened",
        "repository": {
            "full_name": "taichi-dev/taichi"
        },
        "pull_request": {
            "id": 841657847,
            "number": 4224,
            "title": "[lang] Annotate constants with dtype without casting."
        }
    }
    os.environ["GH_EVENT"] = json.dumps(event)
    main()


if __name__ == '__main__':
    main()
