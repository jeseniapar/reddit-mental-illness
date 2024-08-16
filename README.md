# reddit-mental-illness
Predicting suicidal language from Reddit web scraping (praw api)

(1) I scraped a buch of posts from reddit using random generators to manually pick users and their first 20 posts. This created a testing set.

(2) I scraped posts from subreddits that displayed clear mental health issues or suicidal language and identified as either Mental Health, Suicidal or Neither. This created a training set.

(3) I trained a model to parse and identify suicidal language in the original 3000 posts and identify which displayed (a) solely mental health issues (b) suicidal languge or (c) neither. 

(4) Finally completed some statistical analysis on the frequency and types of posts.
