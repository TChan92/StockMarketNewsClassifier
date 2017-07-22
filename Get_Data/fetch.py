import praw
import pandas as pd
import time
import datetime


def check_type(s):
	if type(s) != str:
		return s.encode('ascii', 'replace')
	return s


def create_header(num_headlines):
	return "Date," + "Label," + str(','.join(["Top" + str(i) for i in range(1, num_headlines + 1)]) + "\n")


def date_to_timestamp(date_string):
	return int(time.mktime(datetime.datetime.strptime(date_string, "%Y-%m-%d-%H:%M:%S").timetuple()))


def get_reddit():
	reddit = praw.Reddit(client_id='UtElOjElipazeg',
						 client_secret='HMlvQvokBL6fzHe0mF6aV4qtOZ8',
						 password='alottafish',
						 user_agent='nlp-bot-2017',
						 username='nlp-bot-2017')
	return reddit


# 'timestamp:1420027200..1420070400
def subreddit_search(subreddit, search_string, num_headlines):
	return subreddit.search(search_string, sort='top', limit=num_headlines, syntax='cloudsearch')


'''
CALLING CODE
'''
SUBREDDIT = "stocks"
NUM_HEADLINES = 10

subreddit = get_reddit().subreddit(SUBREDDIT)
data = pd.read_csv('../data/Combined_News_DJIA.csv')
dates = list(data["Date"])
labels = list(data["Label"])
incorrectly_formatted = []
with open("../data/Combined_" + SUBREDDIT + ".csv", 'w') as f:
	f.write(create_header(NUM_HEADLINES))
	for i in range(0, len(dates)):
		d1 = date_to_timestamp(dates[i] + "-00:00:00")
		d2 = date_to_timestamp(dates[i] + "-23:59:00")
		query_string = "timestamp:" + str(d1) + ".." + str(d2)
		time.sleep(1)
		data_row = [dates[i], str(labels[i])]
		for submission in subreddit_search(subreddit, query_string, NUM_HEADLINES):
			for ch in [",", "\n", "\t", "\r"]:
				submission.title = submission.title.replace(ch, "")
			data_row.append(submission.title)
		if (len(data_row) == 2 + NUM_HEADLINES):
			data_row = check_type(','.join(data_row) + "\n")
			f.write(data_row)
		else:
			pass
		print i
f.close()
