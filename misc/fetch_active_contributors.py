import requests

def fetch_active_contributors():
    per_page = 100
    contributors_json = requests.get(f'https://api.github.com/repos/taichi-dev/taichi/contributors?per_page={per_page}').json()
    contributors = []
    print(contributors_json)
    for c in contributors_json:
        contributors.append(c['login'])
        
    assert len(contributors) < per_page, 'Please extend the program to handle >= 100 contributors'
    
    print(len(contributors), contributors)
    
    counter = {}
    
    
    irregular_records = set()
    
    page = 1
    eof = False
    while not eof:
        # Note: for some reason the 'page' argument is 1-based
        commits_json = requests.get(f'https://api.github.com/repos/taichi-dev/taichi/commits?per_page={per_page}&page={page}').json()
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
        
        
    for login, contrib in sorted(list(counter.items()), key=lambda rec: -rec[1]):
        print(f'- [{login}](https://github.com/{login}/)    {contrib}')
        
    print(irregular_records)
    

if __name__ == '__main__':
    fetch_active_contributors()
