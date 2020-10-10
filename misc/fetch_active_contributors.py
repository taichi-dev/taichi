import requests


def print_active_contributors():
    api_prefix = 'https://api.github.com/repos/taichi-dev/taichi'
    per_page = 100

    contributors = []

    page = 1
    while True:
        contributors_json = requests.get(
            f'{api_prefix}/contributors?per_page={per_page}&page={page}').json(
            )

        for c in contributors_json:
            contributors.append(c['login'])

        if len(contributors_json) == 0:
            break
        page += 1

    print(len(contributors), contributors)

    counter = {}

    irregular_records = set()

    page = 1
    eof = False
    while not eof:
        # Note: for some reason the 'page' argument is 1-based
        commits_json = requests.get(
            f'{api_prefix}/commits?per_page={per_page}&page={page}').json()
        for c in commits_json:
            date = c['commit']['committer']['date']
            try:
                author = c['author']['login']
            except:
                irregular_records.add(c['commit']['author']['email'])
                continue
            if int(date[:4]) < 2020:
                eof = True
                break
            print(date, author)
            if author in contributors:
                counter[author] = counter.get(author, 0) + 1

        print('---')
        page += 1

    for login, contrib in sorted(list(counter.items()),
                                 key=lambda rec: -rec[1]):
        print(f'- [{login}](https://github.com/{login}/)    {contrib}')

    print('Irregular records:')
    print(irregular_records)


if __name__ == '__main__':
    print_active_contributors()
