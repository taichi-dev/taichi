#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <map>
#include <list>

using namespace std;

// this function takes a line that may contain a name and/or email address,
// and returns just the name, while fixing the "bad cases".
std::string contributor_name(const std::string& line)
{
  string result;

  // let's first take care of the case of isolated email addresses, like
  // "user@localhost.localdomain" entries
  if(line.find("markb@localhost.localdomain") != string::npos)
  {
    return "Mark Borgerding";
  }

  if(line.find("kayhman@contact.intra.cea.fr") != string::npos)
  {
    return "Guillaume Saupin";
  }

  // from there on we assume that we have a entry of the form
  // either:
  //   Bla bli Blurp
  // or:
  //   Bla bli Blurp <bblurp@email.com>
  
  size_t position_of_email_address = line.find_first_of('<');
  if(position_of_email_address != string::npos)
  {
    // there is an e-mail address in <...>.
    
    // Hauke once committed as "John Smith", fix that.
    if(line.find("hauke.heibel") != string::npos)
      result = "Hauke Heibel";
    else
    {
      // just remove the e-mail address
      result = line.substr(0, position_of_email_address);
    }
  }
  else
  {
    // there is no e-mail address in <...>.
    
    if(line.find("convert-repo") != string::npos)
      result = "";
    else
      result = line;
  }

  // remove trailing spaces
  size_t length = result.length();
  while(length >= 1 && result[length-1] == ' ') result.erase(--length);

  return result;
}

// parses hg churn output to generate a contributors map.
map<string,int> contributors_map_from_churn_output(const char *filename)
{
  map<string,int> contributors_map;

  string line;
  ifstream churn_out;
  churn_out.open(filename, ios::in);
  while(!getline(churn_out,line).eof())
  {
    // remove the histograms "******" that hg churn may draw at the end of some lines
    size_t first_star = line.find_first_of('*');
    if(first_star != string::npos) line.erase(first_star);
    
    // remove trailing spaces
    size_t length = line.length();
    while(length >= 1 && line[length-1] == ' ') line.erase(--length);

    // now the last space indicates where the number starts
    size_t last_space = line.find_last_of(' ');
    
    // get the number (of changesets or of modified lines for each contributor)
    int number;
    istringstream(line.substr(last_space+1)) >> number;

    // get the name of the contributor
    line.erase(last_space);    
    string name = contributor_name(line);
    
    map<string,int>::iterator it = contributors_map.find(name);
    // if new contributor, insert
    if(it == contributors_map.end())
      contributors_map.insert(pair<string,int>(name, number));
    // if duplicate, just add the number
    else
      it->second += number;
  }
  churn_out.close();

  return contributors_map;
}

// find the last name, i.e. the last word.
// for "van den Schbling" types of last names, that's not a problem, that's actually what we want.
string lastname(const string& name)
{
  size_t last_space = name.find_last_of(' ');
  if(last_space >= name.length()-1) return name;
  else return name.substr(last_space+1);
}

struct contributor
{
  string name;
  int changedlines;
  int changesets;
  string url;
  string misc;
  
  contributor() : changedlines(0), changesets(0) {}
  
  bool operator < (const contributor& other)
  {
    return lastname(name).compare(lastname(other.name)) < 0;
  }
};

void add_online_info_into_contributors_list(list<contributor>& contributors_list, const char *filename)
{
  string line;
  ifstream online_info;
  online_info.open(filename, ios::in);
  while(!getline(online_info,line).eof())
  {
    string hgname, realname, url, misc;
    
    size_t last_bar = line.find_last_of('|');
    if(last_bar == string::npos) continue;
    if(last_bar < line.length())
      misc = line.substr(last_bar+1);
    line.erase(last_bar);
    
    last_bar = line.find_last_of('|');
    if(last_bar == string::npos) continue;
    if(last_bar < line.length())
      url = line.substr(last_bar+1);
    line.erase(last_bar);

    last_bar = line.find_last_of('|');
    if(last_bar == string::npos) continue;
    if(last_bar < line.length())
      realname = line.substr(last_bar+1);
    line.erase(last_bar);

    hgname = line;
    
    // remove the example line
    if(hgname.find("MercurialName") != string::npos) continue;
    
    list<contributor>::iterator it;
    for(it=contributors_list.begin(); it != contributors_list.end() && it->name != hgname; ++it)
    {}
    
    if(it == contributors_list.end())
    {
      contributor c;
      c.name = realname;
      c.url = url;
      c.misc = misc;
      contributors_list.push_back(c);
    }
    else
    {
      it->name = realname;
      it->url = url;
      it->misc = misc;
    }
  }
}

int main()
{
  // parse the hg churn output files
  map<string,int> contributors_map_for_changedlines = contributors_map_from_churn_output("churn-changedlines.out");
  //map<string,int> contributors_map_for_changesets = contributors_map_from_churn_output("churn-changesets.out");
  
  // merge into the contributors list
  list<contributor> contributors_list;
  map<string,int>::iterator it;
  for(it=contributors_map_for_changedlines.begin(); it != contributors_map_for_changedlines.end(); ++it)
  {
    contributor c;
    c.name = it->first;
    c.changedlines = it->second;
    c.changesets = 0; //contributors_map_for_changesets.find(it->first)->second;
    contributors_list.push_back(c);
  }
  
  add_online_info_into_contributors_list(contributors_list, "online-info.out");
  
  contributors_list.sort();
  
  cout << "{| cellpadding=\"5\"\n";
  cout << "!\n";
  cout << "! Lines changed\n";
  cout << "!\n";

  list<contributor>::iterator itc;
  int i = 0;
  for(itc=contributors_list.begin(); itc != contributors_list.end(); ++itc)
  {
    if(itc->name.length() == 0) continue;
    if(i%2) cout << "|-\n";
    else cout << "|- style=\"background:#FFFFD0\"\n";
    if(itc->url.length())
      cout << "| [" << itc->url << " " << itc->name << "]\n";
    else
      cout << "| " << itc->name << "\n";
    if(itc->changedlines)
      cout << "| " << itc->changedlines << "\n";
    else
      cout << "| (no information)\n";
    cout << "| " << itc->misc << "\n";
    i++;
  }
  cout << "|}" << endl;
}
