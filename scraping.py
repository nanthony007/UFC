import pandas as pd
import requests
import re
import string
from bs4 import BeautifulSoup
from operator import itemgetter
import pickle
from datetime import datetime


def combine(new_data):
    """Adds new data to existing csv file
    
    Arguments:
        new_data {[df]} -- [new df to add to main_df]
    """    
        
    main_df = pd.read_csv('UFC_stats.csv')
    main_df.append(new_data, ignore_index=True)
    main_df.to_csv('UFC_stats.csv')
    return


def generate_fighter_urls():
    """Generates a-z urls to parse for fighters, returns list of links
    
    Returns:
        list of urls -- each url for each letter on the fighter search page
    """    
    fighter_url_base = 'http://www.ufcstats.com/statistics/fighters?char='
    return [(fighter_url_base + letter + '&page=all')
            for letter in string.ascii_lowercase]


def get_fighter_links(alphabet_links):
    """Gets fighter profile links from each alphabet page
    
    Arguments:
        alphabet_links {list} -- list of urls for fighter pages from generate_fighter_urls
    
    Returns:
        list -- fighter profile urls
    """    
    # local list
    fighter_profile_links = []
    for link in alphabet_links:
        response = requests.get(link)
        soup = BeautifulSoup(response.text, features='html.parser')
        fighter_profile_links.extend([row.get('href')
                                      for row in soup.select('td.b-statistics__table-col a')])

    return list(set(fighter_profile_links))  # ensures no duplicates


def parse_fighter_profile(link):
    """Parses fighter profile for demographic info.
    
    Arguments:
        link {list of links} -- list of links from get_fighter_links
    
    Returns:
        name -- name of fighter
        info_list -- demographic info on fighter
    """    
    response = requests.get(link)
    soup = BeautifulSoup(response.text, features='html.parser')
    # parses name out of html
    name = [x.text.strip() for x in soup.select('div h2 span.b-content__title-highlight')][0]
    # gets box with demographic info
    demographics = soup.select(
        'div.b-list__info-box.b-list__info-box_style_small-width.js-guide ul li')

    info_list = []
    # all stats get checked to see if there is valid data based on length

    # height in inches
    if len(demographics[0].text.split()) > 2:
        info_list.append(int(demographics[0].text.split()[-2].strip('\'')) * 12 + int(
            demographics[0].text.split()[-1].strip('\"')))
    else:
        info_list.append(None)

    # weight in pounds
    if len(demographics[1].text.split()) > 2:
        info_list.append(int(demographics[1].text.split()[-2]))
    else:
        info_list.append(None)

    # reach in inches
    info_list.append(int(demographics[2].text.split()[-1].replace('"', '')) if '"' in demographics[
        2].text.split()[-1] else None)

    # DOB in datetime format
    if len(demographics[4].text.split(':')[-1].strip()) > 2:
        info_list.append(
            datetime.strptime(demographics[4].text.split(':')[-1].strip(), '%b %d, %Y'))
    else:
        info_list.append(None)

    # fighter profile link
    info_list.append(response.url)

    return name, info_list


def filter_profile_links(fighter_page_links):
    """Filters fighter profile links and calls parse_fighter_profile.
    
    Arguments:
        fighter_page_links {list} -- list of profile links for all fighters
    
    Returns:
        int -- count on new fighter profiles added
    """    
    """"""

    # load existing fighter dictionary
    with open('fighters.pkl', 'rb') as f:
        # uses default dict collection
        # makes easier to add new and update duplicates
        fighter_info = pickle.load(f)

    new_fighter_counts = 0
    for link in fighter_page_links:
        # checks if link is in dict already
        # i.e. checks if profile has been scraped previously
        if any(link in v for v in fighter_info.values()):
            # pass for now
            # could need to re-scrape for newer info if profiles are updated
            pass
        else:
            # parse page for fighter info
            new_fighter = parse_fighter_profile(link)  # returns name, info

            # adds new fighter to dict
            fighter_info[new_fighter[0]] = new_fighter[1]
            new_fighter_counts += 1

    # saves changes
    with open('fighters.pkl', 'wb') as f:
        pickle.dump(fighter_info, f)

    return new_fighter_counts


def double_fights(fight_list):
    """Doubles fights for analysis purposes, returns doubled fight list.
    
    Arguments:
        fight_list {list of fight data} -- from parse_fight_pages
    
    Returns:
        doubled fight lists -- doubled fights for analysis purposes
    """

    doubled_list = [[fight[1], fight[0], fight[3], fight[2], fight[5], fight[4], fight[8], fight[9],
                     fight[6], fight[7], fight[12], fight[13], fight[10], fight[11], fight[16],
                     fight[17], fight[14], fight[15], fight[19], fight[18], fight[20], fight[21],
                     fight[23], fight[22], fight[26], fight[27], fight[24], fight[25], fight[30],
                     fight[31], fight[28], fight[29], fight[34], fight[35], fight[32], fight[33],
                     fight[38], fight[39], fight[36], fight[37], fight[42], fight[43], fight[40],
                     fight[41], fight[46], fight[47], fight[44], fight[45], fight[48], fight[49],
                     fight[50]] for fight in fight_list]

    return fight_list + doubled_list


def add_fighter_info(fight_list, new_file_name):
    """Adds fighter info to each fight list, returns data frame and writes to csv.
    
    Arguments:
        fight_list {list} -- doubled fight list from double_fights
        new_file_name {string} -- name of new file
    
    Returns:
        df -- new fightdata df
    """    
    # load existing fighter dictionary
    with open('fighters.pkl', 'rb') as f:
        # uses default dict collection
        # makes easier to add new and update duplicates
        fighter_info = pickle.load(f)

    # adds fighter info list after each fighter's name
    for fight in fight_list:
        if fight[0] in fighter_info.keys() and fight[1] in fighter_info.keys():
            fight.insert(1, fighter_info[fight[0]][0])
            fight.insert(2, fighter_info[fight[0]][1])
            fight.insert(3, fighter_info[fight[0]][2])
            fight.insert(4, fighter_info[fight[0]][3])
            fight.insert(5, fighter_info[fight[0]][4])
            fight.insert(7, fighter_info[fight[6]][0])
            fight.insert(8, fighter_info[fight[6]][1])
            fight.insert(9, fighter_info[fight[6]][2])
            fight.insert(10, fighter_info[fight[6]][3])
            fight.insert(11, fighter_info[fight[6]][4])
        else:
            pass

    labels = ['Fighter1', 'F1_Height', 'F1_Weight', 'F1_Reach', 'F1_DOB', 'F1_profile_url',
              'Fighter2', 'F2_Height', 'F2_Weight', 'F2_Reach', 'F2_DOB', 'F2_profile_url',
              'F1_result', 'F2_result', 'F1_KD', 'F2_KD', 'F1_SS_hit', 'F1_SS_att', 'F2_SS_hit',
              'F2_SS_att', 'F1_totalStrikes_hit', 'F1_totalStrikes_att', 'F2_totalStrikes_hit',
              'F2_totalStrikes_att', 'F1_TD_conv', 'F1_TD_att', 'F2_TD_conv', 'F2_TD_att',
              'F1_Sub', 'F2_Sub', 'F1_pass', 'F2_pass', 'F1_rev', 'F2_rev', 'F1_head_hit',
              'F1_head_att', 'F2_head_hit', 'F2_head_att', 'F1_body_conv', 'F1_body_att',
              'F2_body_conv', 'F2_body_att', 'F1_leg_conv', 'F1_leg_att', 'F2_leg_conv',
              'F2_leg_att', 'F1_distance_conv', 'F1_distance_att', 'F2_distance_conv',
              'F2_distance_att', 'F1_clinch_conv', 'F1_clinch_att', 'F2_clinch_conv',
              'F2_clinch_att', 'F1_ground_conv', 'F1_ground_att', 'F2_ground_conv',
              'F2_ground_att', 'Date', 'Location', 'Attendance']

    df = pd.DataFrame(fight_list, columns=labels)
    df.to_csv(new_file_name, index=False)  # removes index in saving

    return df


def parse_main_page(event_count,
                    url="http://www.ufcstats.com/statistics/events/completed?page=all"):
    """Parses main page and returns links.
    
    Arguments:
        event_count {int} -- number of events to parse
    
    Keyword Arguments:
        url {str} -- homepage link (default: {"http://www.ufcstats.com/statistics/events/completed?page=all"})
    
    Returns:
        list -- list of event links
    """    
    """"""

    response = requests.get(url)

    soup = BeautifulSoup(response.text, features='html.parser')
    table_links = soup.select('tr.b-statistics__table-row td i a')

    linklist = []
    for link in table_links[:event_count]:
        linklist.append(link.get('href'))

    return linklist


def parse_event_pages(linklist):
    """Parses individual event pages for fight links.
    
    Arguments:
        linklist {list} -- event link list from parse_main_page
    
    Returns:
        event_fight_locations -- locations of events
        event_fight_links -- links for fights from event
    """    
    # STILL NEEDS TO GET METHOD OF RESULT EVENTUALLY?

    # local lists
    event_fight_locations = []
    event_fight_links = []

    for link in linklist:
        response = requests.get(link)
        soup = BeautifulSoup(response.text, features='html.parser')

        # get links for event fights
        fight_links = [x.get('data-link') for x in soup.select('tr.js-fight-details-click')]
        event_fight_links.append(fight_links)

        # get location info for event
        location_info = soup.select('li.b-list__box-list-item')

        # getting 3 items from page (date, location, attendance + #fights from above)
        loc_details = [item[0]
                       for item in itemgetter(2, 5, 8)([y.string.strip().split(':')
                                                        for x in location_info for y in x])]

        # rewrite list with python data types
        # replaces missing attendance with None
        event_fight_locations.extend(
            [(datetime.strptime(loc_details[0], '%B %d, %Y'),
             loc_details[1],
             re.sub(',', '', loc_details[2]))
             if len(re.sub(',', '', loc_details[2])) > 0
             else (
                datetime.strptime(loc_details[0], '%B %d, %Y'),
                loc_details[1],
                None,
            )
            ]
        )

    return event_fight_locations, event_fight_links


def parse_fight_pages(location_list, linklist, num_of_events):
    """Parses data from fight pages.
    
    Arguments:
        location_list {list} -- [list of event locations from parse_event_pages]
        linklist {[list]} -- [list of fights in event from parse_event_pages]
        num_of_events {int} -- [number of events parsed]
    
    Returns:
        [list] -- [final list of fight data]
    """    
    # STILL NEEDS TO GET METHOD OF RESULT EVENTUALLY?
    # function list
    final_list = []

    for e in range(num_of_events):  # iterates based on # of events
        for link in linklist[e]:
            # local list
            fight_stats = []

            response = requests.get(link)
            soup = BeautifulSoup(response.text, features='html.parser')

            # gets fighter names from header
            fighters = [x.text.strip() for x in soup.select('h3.b-fight-details__person-name')]
            fight_stats.append(fighters[0])
            fight_stats.append(fighters[1])

            # gets results from header icons
            results = [x.text.strip() for x in soup.select('i.b-fight-details__person-status')]
            fight_stats.append(results[0])
            fight_stats.append(results[1])

            # parse stats from page
            stats_html = soup.find_all('table', attrs={"style": "width: 745px"})

            # remove soup formatting and get raw values
            full_stats = [ss.strip() for s in stats_html
                          for ss in re.split('\n', s.tbody.tr.text.strip()) if len(ss.strip()) > 0]

            # zips together even and odd lists then combines into list pairs
            grouped_stats = [list(x) for x in list(zip(full_stats[0::2], full_stats[1::2]))]

            # pops the % stats and names
            unwanted_indices = [0, 2, 4, 7, 7, 7]
            for i in unwanted_indices:
                grouped_stats.pop(i)

            # FROM HERE DOWN CAN BE IMPROVED!!!
            # uses regex to split 'of' string stats
            for i, pair in enumerate(grouped_stats):
                grouped_stats[i] = [re.split(r'\sof\s', item) for item in pair]

            # flattens nested lists from above steps
            for i, pair in enumerate(grouped_stats):
                for z, item in enumerate(pair):
                    for y, last in enumerate(item):
                        grouped_stats[i][z][y] = int(last)

            # further flattens list
            new_groups = [y for x in grouped_stats for y in x]

            # final list flatten
            # checks if list only has 1 stat or 2 then appends each to new (flat) list
            mini_list = []
            for i, subgroup in enumerate(new_groups):
                if len(subgroup) == 1:
                    mini_list.append(subgroup[0])
                else:
                    mini_list.append(subgroup[0])
                    mini_list.append(subgroup[1])

            fight_stats.extend(mini_list)  # adds stats
            fight_stats.extend(location_list[e])  # adds location
            print(fight_stats)  # can remove if want... prints each fight per iteration
            final_list.append(fight_stats)
    return final_list


def main_call(events_count):  # param should be int
    """Runs main program, calls all other functions."""
    fighter_alphabet_links = generate_fighter_urls()
    print('Step 1 Complete --- Generated Alphabet Links')
    fighter_links = get_fighter_links(fighter_alphabet_links)
    print('Step 2 Complete --- Generated Fighter Links')
    new_fighters = filter_profile_links(fighter_links)
    print('Step 3 Complete --- Parsed ' + str(new_fighters) + ' New Profiles')
    main_page_links = parse_main_page(events_count)
    print('Step 4 Complete --- Parsed Main Page')
    event_page_links = parse_event_pages(main_page_links)
    print('Step 5 Complete --- Parsed Event Pages')
    fight_page_data = parse_fight_pages(event_page_links[0], event_page_links[1], events_count)
    print('Step 6 Complete --- Parsed Fight Pages')
    doubled_fight_data = double_fights(fight_page_data)
    print('Step 7 Complete --- Doubled Fight List')
    finished_new_data = add_fighter_info(doubled_fight_data, 'UFC_stats_add1.csv')
    verify = input('Add to main csv file? [y/n]')
    if verify == 'y':
        combine(finished_new_data)  # returns True
        print('New data added')
    else:
        print('New data not added')


if __name__ == "__main__":
    main_call(11)
