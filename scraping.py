# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython

#%%
import pandas as pd
import numpy as np
import requests
import re
import string
from bs4 import BeautifulSoup
from requests_futures.sessions import FuturesSession
from operator import itemgetter

# everything needs modularization for each page 
# can get all EVENTS from one page
# then each event has multiple fights (each with a page)... 
# use a function in a loop for this

# one function for getting from events page to individual fight page
# another function to scrape fight data


#%%
url = "http://www.ufcstats.com/statistics/events/completed?page=all"
response = requests.get(url)
print(response.url)

soup = BeautifulSoup(response.text)
tablelinks = soup.select('tr.b-statistics__table-row td i a')

linklist = []
for link in tablelinks:
    linklist.append(link.get('href'))
    
len(linklist)


#%%
linklist[:2]


#%%
len(linklist)


#%%
demolist = []
factlist = []
tablelinks = []
eventfighterslist = []

parse(requests.get(linklist[1]))


#%%
def parse(response):
    if response.status_code != 200:
        print(response.status_code)
    soup = BeautifulSoup(response.text)
    l = soup.select('tr.js-fight-details-click')
    tablelinks.append(l)
    e = [x.text.strip() for x in soup.select('p.b-fight-details__table-text a') if len(x.text.strip()) > 5]
    eventfighterslist.append(list(zip(*[iter(e)] * 2)))
    factlist.append(soup.select('p.b-fight-details__table-text'))
    fcount.append(len(l))
    demolist.append(soup.select('li.b-list__box-list-item'))
    return response.status_code

def scrape(index, future):
    if future.status_code != 200:
        return index
    else:
        parse(future)
        return future.status_code

def get(url):
    try:
        return requests.get(url)
    except Exception:
        # sleep for a bit in case that helps
        time.sleep(2)
        # try again
        return get(url)


#%%
from requests_futures.sessions import FuturesSession

session = FuturesSession(max_workers=5)

# takes a few minutes
demolist = []
factlist = []
tablelinks = []
eventfighterslist = []
fcount = []

futures = []

for i, url in enumerate(linklist):
    x = get(url)
    print(i, x.status_code)
    futures.append(x)
    


#%%
Z = [scrape(i, future) for i, future in enumerate(futures)]


#%%
badrounds = [x for x in Z if x != 200]
print(len(badrounds))
badrounds


#%%
#retry
ff = [session.get(linklist[i]) for i in badrounds]

retry = [scrape(i, f) for i, f in enumerate(ff)]


#%%
print(len(demolist))
print(len(factlist))
print(len(tablelinks))
print(len(eventfighterslist))
print(len(fcount))


#%%
fcount[0]


#%%
fightresults = []

for c, fight in zip(fcount[:2], factlist[:2]):
    fr = [i.text.strip() for i in fight]
    for f in fr:
        print(f, len(f))
    frr = [fr[i:i+c] for i in range(0, len(fr), c)]
    for f in frr:
        print(f)
    #fightresults.append(fr)


#%%
[factlist[i:i + 16] for i in range(0, len(factlist), 16)][0][0]


#%%
len(fightresults)


#%%
fightresults[1]


#%%
fightinfo = []
for item in demolist:
    fightinfo.append([y.strip() for x in item for y in re.sub('\n', '', x.text.strip()).split(':')])


#%%
fightinfo[2]


#%%
eventdata = []
for i in range(0, len(fightinfo)):
    a = list(itemgetter(1, 3, 5)(fightinfo[i]))
    b = list(eventfighterslist[i])
    a.append(b)
    eventdata.append(a)


#%%
cols = ['date', 'location', 'attendance', '#fights']
eventdata[0]


#%%
from datetime import datetime

eventdata2 = [(datetime.strptime(event[0], '%B %d, %Y'), event[1], re.sub(',', '', event[2]), event[-1]) 
 if len(re.sub(',', '', event[2])) > 0 
 else (datetime.strptime(event[0], '%B %d, %Y'), event[1], None, event[-1])
 for event in eventdata]


#%%
eventdata2[0]


#%%
eventdata2[0][:-1]


#%%
fightlinks = []

for link in tablelinks:
    for item in link:
        fightlinks.append(item.get('data-link'))


#%%
len(fightlinks)


#%%
fightlinks[:2]


#%%
get_ipython().system(' get fight data itself !')


#%%
def scrape2(index, future):
    x = future.result()
    if x.status_code != 200:
        print(index, x.status_code)
        return index
    else:
        parse2(x)
        return x.status_code
    
def parse2(response):
    soup = BeautifulSoup(response.text)
    fightsHTML = soup.select('h3.b-fight-details__person-name')
    fight_fighters = [f.text.strip() for f in fightsHTML]
    resultsHTML = soup.select('i.b-fight-details__person-status')
    fight_results = [r.text.strip() for r in resultsHTML]
    statsHTML = soup.find_all('table', attrs={"style": "width: 745px"})
    if len(statsHTML) == 0:
        pass
    else:
        full_stats = [ss.strip() for s in statsHTML for ss in re.split('\n', s.tbody.tr.text.strip()) if len(ss.strip()) > 0]
        grouped_stats = list(zip(full_stats[0::2], full_stats[1::2]))
        grouped_stats = [list(x) for x in grouped_stats]

        for i in notneeded:
            grouped_stats.pop(i)
        for i, pair in enumerate(grouped_stats):
            grouped_stats[i] = [re.split(r'\sof\s', i) for i in pair]
        for i, pair in enumerate(grouped_stats):
            for z, item in enumerate(pair):
                for y, last in enumerate(item):
                    grouped_stats[i][z][y] = int(last)



        new_groups = [y for x in grouped_stats for y in x]
        minilist = []
        for i, subgroup in enumerate(new_groups):
            if len(subgroup) == 1:
                minilist.append(subgroup[0])
            else:
                minilist.append(subgroup[0])
                minilist.append(subgroup[1])

        fs1 = fight_fighters[0]
        fs2 = fight_fighters[1]
        rs1 = fight_results[0]
        rs2 = fight_results[1]

        sublist = [fs1, fs2, rs1, rs2]
        sublist.extend(minilist)
        full_fight_stats.append(sublist)
    return response.status_code


#%%
session = FuturesSession(max_workers=5)

# takes a while
fightsHTML = []
resultsHTML = []
statsHTML = []

full_fight_stats = []
notneeded = [0, 2, 4, 7, 7, 7]

futures = [session.get(url) for url in fightlinks]


#%%
fightlinks[5049]


#%%
ZZ = []
for i, future in enumerate(futures):
    ZZ.append(scrape2(i, future))


#%%
len(full_fight_stats)


#%%
badrounds = [x for x in ZZ if x != 200]
len(badrounds)


#%%
#retry2
ff = [session.get(fightlinks[i]) for i in badrounds]

retry = [scrape2(i, f) for i, f in enumerate(ff)]


#%%
len(full_fight_stats)


#%%
len(futures)


#%%
# fighter data
fighter_url_base = 'http://www.ufcstats.com/statistics/fighters?char='

fighter_letters_urls = []
for letter in string.ascii_lowercase:
    fighter_letters_urls.append(fighter_url_base + letter + '&page=all')
    
fighter_letters_urls[0]


#%%
from requests_futures.sessions import FuturesSession

session = FuturesSession(max_workers=5)
fighter_links = []

futures = [session.get(url) for url in fighter_letters_urls]
for future in futures:
    soup = BeautifulSoup(future.result().content)
    rows = soup.select('td.b-statistics__table-col a') 
    for row in rows:
        fighter_links.append(row.get('href'))
        
fighter_links = list(set(fighter_links))


#%%
len(fighter_links)


#%%
fighter_links[:3]


#%%
def scrape(index, future):
    x = future.result()
    print(index, x.status_code)
    if x.status_code != 200:
        return index
    else:
        parse(x)
        return x.status_code

def parse(future):
    soup = BeautifulSoup(future.content)
    name = soup.select('div h2 span.b-content__title-highlight')
    demographics = soup.select('div.b-list__info-box.b-list__info-box_style_small-width.js-guide ul li')
    
    hwrb = []
    # height
    if len(demographics[0].text.split()) > 2:
        hwrb.append(int(demographics[0].text.split()[-2].strip('\'')) * 12 + int(demographics[0].text.split()[-1].strip('\"')))
    else:
        hwrb.append(None) 
        
    # weight
    if len(demographics[1].text.split()) > 2:
        hwrb.append(int(demographics[1].text.split()[-2]))
    else:
        hwrb.append(None)
    
    # reach
    hwrb.append(int(demographics[2].text.split()[-1].replace('"', '')) if '"' in demographics[2].text.split()[-1] else None)
    
    # DOB
    if len(demographics[4].text.split(':')[-1].strip()) > 2:
        hwrb.append(datetime.strptime(demographics[4].text.split(':')[-1].strip(), '%b %d, %Y'))
    else:
        hwrb.append(None) 
    
    # fighter profile link
    hwrb.append(future.url)
    
    # create fighter dictionary entry
    fighter_info[[i.text.strip() for i in name][0]] = hwrb


#%%
# takes a few minutes
session = FuturesSession(max_workers=5)

# takes a while

futures = [session.get(url) for url in fighter_links]

# futures = []
# for i, url in enumerate(fighter_links):
#     futures.append(requests.get(url))
#     if i % 100 == 0:
#         print(i)
# #futures = [requests.get(url) for url in fighter_links]


#%%
len(futures)


#%%
cats = ['Name', 'Height', 'Weight', 'Reach', 'DOB', 'profilelink']

from collections import defaultdict
import time
from datetime import datetime
from collections import defaultdict
from IPython.display import clear_output

fighter_info = defaultdict()
Z = []

for i, future in enumerate(futures):
    Z.append(scrape(i, future))
#clear_output()


#%%
from collections import Counter

Counter(Z)


#%%
badrounds = [x for x in Z if x != 200]
len(badrounds)

#retry
ff = [session.get(fighter_links[i]) for i in badrounds]

retry = [scrape(i, f) for i, f in enumerate(ff)]


#%%
badrounds2 = [x for x in retry if x != 200]
len(badrounds2)

#retry
ff2 = [session.get(fighter_links[i]) for i in badrounds2]

retry2 = [scrape(i, f) for i, f in enumerate(ff2)]


#%%
len(fighter_info)


#%%
# use with caution
for k, v in fighter_info.items():
    print(k, v)


#%%
len(fighter_info)


#%%
fighter_info['Ben Askren']


#%%
import pickle


#%%
with open('fighters.pkl', 'wb') as f:
    pickle.dump(fighter_info, f)


#%%
with open('events.pkl', 'wb') as e:
    pickle.dump(eventdata2, e)


#%%
with open('full_fight_stats.pkl', 'wb') as ffs:
    pickle.dump(full_fight_stats, ffs)


#%%
with open('fightlinks.pkl', 'wb') as fl:
    pickle.dump(fightlinks, fl)


#%%
with open('fightresults.pkl', 'wb') as fr:
    pickle.dump(fighresults, fr)


#%%
with open('fightresults.pkl', 'rb') as fr:
    fightresults = pickle.load(fighresults)


#%%
with open('fightlinks.pkl', 'rb') as fl:
    fightlinks = pickle.load(fl)


#%%
with open('fighters.pkl', 'rb') as f:
    fighter_info = pickle.load(f)


#%%
with open('events.pkl', 'rb') as e:
    eventdata2 = pickle.load(e)


#%%
with open('full_fight_stats.pkl', 'rb') as ffs:
    full_fight_stats = pickle.load(ffs)


#%%



#%%



#%%
for event in eventdata2:
    for i, fpair in enumerate(event[-1]):
        if fpair == ('draw', 'draw'):
            event[-1].pop(i)


#%%
[e for e in eventdata2 if ('draw', 'draw') in e[-1]] # check for above cell


#%%
for event in eventdata2:
    fpairs = event[-1]
    for pair in fpairs:
        for i, fight in enumerate(full_fight_stats): 
            if tuple([fight[0], fight[1]]) == pair or tuple([fight[1], fight[0]]) == pair:
                if len(fight) == 48:
                    fight.extend(event[:-1])
                    break
#                 elif len(fight) == 51:
#                     full_fight_stats[i] = fight[:-3] + list(event[:-1])

#     for x, fight in enumerate(full_fight_stats):
#         fs1 = tuple([fight[0], fight[1]])
#         fs2 = tuple([fight[1], fight[0]])
#         if fs1 in fpairs or fs2 in fpairs:
#             if len(fight) == 48:
#                 fight.extend(event[:-1])
#             elif len(fight) == 51:
#                 fight = fight[:-3]+event[-1]


#%%
len([(i,len(l)) for i,l in enumerate(full_fight_stats) if len(l) != 51])


#%%
full_fight_stats = [fight for fight in full_fight_stats if len(fight) == 51]


#%%
doubled = [[fight[1], fight[0], fight[3], fight[2], fight[5], fight[4], fight[8], fight[9], fight[6], fight[7], fight[12], fight[13],
 fight[10], fight[11], fight[16], fight[17], fight[14], fight[15], fight[19], fight[18], fight[20], fight[21], fight[23], fight[22],
  fight[26], fight[27], fight[24], fight[25], fight[30], fight[31], fight[28], fight[29], fight[34], fight[35], fight[32], fight[33], 
  fight[38], fight[39], fight[36], fight[37], fight[42], fight[43], fight[40], fight[41], fight[46], fight[47], fight[44], fight[45],
  fight[48], fight[49], fight[50]] for fight in full_fight_stats]


#%%
FIGHT = full_fight_stats + doubled
len(FIGHT)


#%%
for f in FIGHT:
    if f[0] == 'Byron Bloodworth' or f[1] == 'Byron Bloodworth':
        print(f)


#%%
for f in FIGHT:
    if f[0] in fighter_info.keys() and f[1] in fighter_info.keys():
        f.insert(1, fighter_info[f[0]][0])
        f.insert(2, fighter_info[f[0]][1])
        f.insert(3, fighter_info[f[0]][2])
        f.insert(4, fighter_info[f[0]][3])
        f.insert(5, fighter_info[f[0]][4])
        f.insert(7, fighter_info[f[6]][0])
        f.insert(8, fighter_info[f[6]][1])
        f.insert(9, fighter_info[f[6]][2])
        f.insert(10, fighter_info[f[6]][3])
        f.insert(11, fighter_info[f[6]][4])
    else: 
        pass


#%%
fight = [f for f in FIGHT if len(f) == 61]
len(fight)


#%%
labels = ['Fighter1', 'F1_Height', 'F1_Weight', 'F1_Reach', 'F1_DOB', 'F1_profile_url', 
          'Fighter2','F2_Height', 'F2_Weight', 'F2_Reach', 'F2_DOB', 'F2_profile_url',
          'F1_result', 'F2_result', 'F1_KD', 'F2_KD', 'F1_SS_hit', 'F1_SS_att', 'F2_SS_hit', 'F2_SS_att',
          'F1_totalStrikes_hit', 'F1_totalStrikes_att', 'F2_totalStrikes_hit', 'F2_totalStrikes_att', 'F1_TD_conv', 'F1_TD_att', 
           'F2_TD_conv', 'F2_TD_att', 'F1_Sub', 'F2_Sub', 'F1_pass', 'F2_pass', 'F1_rev', 'F2_rev', 
          'F1_head_hit', 'F1_head_att', 'F2_head_hit', 'F2_head_att', 'F1_body_conv', 'F1_body_att', 'F2_body_conv', 'F2_body_att', 
          'F1_leg_conv', 'F1_leg_att', 'F2_leg_conv', 'F2_leg_att', 'F1_distance_conv', 'F1_distance_att', 'F2_distance_conv', 'F2_distance_att',
          'F1_clinch_conv', 'F1_clinch_att', 'F2_clinch_conv', 'F2_clinch_att', 'F1_ground_conv', 'F1_ground_att', 'F2_ground_conv', 'F2_ground_att',
          'Date', 'Location', 'Attendance']
len(labels)


#%%
import pandas as pd
df = pd.DataFrame(fight, columns=labels)
print(len(df))
df.head()


#%%
df.to_csv('UFCstats.csv', index=False) # keyword arg removes index in saving


#%%
df = pd.read_csv('UFCstats.csv')
df.Attendance.isnull()


#%%
df[df.Fighter1=='Daniel Cormier'][df.Fighter2 == 'Stipe Miocic']


#%%
df[df.isnull().any(axis=1)]


#%%
df[df.Fighter1 == 'Daniel Cormier'][df.Fighter2=='Stipe Miocic'][df.Date == '2018-07-07']


#%%
df[df.Fighter1 == 'Stipe Miocic'][df.Fighter2 == 'Daniel Cormier'][df.Date == '2018-07-07']


#%%



