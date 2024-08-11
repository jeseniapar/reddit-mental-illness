import praw
import pandas as pd

## First collecting Schizophrenic patient data

reddit = praw.Reddit(
    client_id="EgYhtuW9L0CD3LBIgvkvrw",
    client_secret="BeUvMfWktpPrGVQK5myrTaw8scgnLA",
    user_agent="Schiz/suicide correlation by /u/NecessaryClassic4135"
)

usernames = ["sunfloras", "No-Molasses-2247", 'Repulsive_Ring_2309', 'helsdog', 'Mental_Draft_', "Icy-Artichoke3305", "-Baguette_", "Basil-the-bagel", "RenivaMa", "aloafaloft", "vacantxwhxre", "x37h4n", "venomang", "Rude-Ad-8051", "drowsyneon", "Independent_Pain3944", "MuchAd8491", "Great-FOXX", "PsychieLeaks", "booshweasel", "pringusdingus6600", "Rude_Economist9099", "InterestingKiwi5004", "axil8888", "Low-Tailor-4062", "ToughLuckLukeLuke", "kirs1132", "latudalithium", "logscaledtree", "trashaccountturd", "blahblahlucas", "schizofuqface", "Arbitrary-Signal", "Outside-Age5073", "arpanetas", "ResurgentClusterfuck", "Roxanngreen83", "Rebephrenic_", "NegativeEngine", "Cute-Avali", "Successful_Room_3576", "Oosteocyte", "sas0002", "BitersAnon", "darkfireice", "deamon83", "Warm-Inflation-5734", "Jaye-Fern", "cosmiccatface", "trashaccountturd"]

data = []
for username in usernames:
    # Get reddit user as object
    user = reddit.redditor(username)
    
    # get their top 20 posts (most recent)
    submissions = user.submissions.new(limit=20)
    
    # loop through and append the data
    for submission in submissions:
        data.append({
            "Username": username,
            "Post Text": submission.selftext
        })


# convert dictionaries into a dataframe
study = pd.DataFrame(data)
study['Diagnosis'] = "SCH"
study

## Second, collecting mental health users

from prawcore.exceptions import NotFound

reddit = praw.Reddit(
    client_id="EgYhtuW9L0CD3LBIgvkvrw",
    client_secret="BeUvMfWktpPrGVQK5myrTaw8scgnLA",
    user_agent="Schiz/suicide correlation by /u/NecessaryClassic4135"
)

usernames = ["UtopianPariah", "iloveredditrabbit", "LisKoz1989", "joeym412", "SillyDot3305", "J3rk_d5", "HauntingWeather3289", "Darkahrii", "GlitterCoconut", "CarSubstantial5238", "angelpawsz", "BottleOfFlesh", "goldenhobisediting", "Keny376", "Strict-Impression650", "eaglesfan700", "Crowgogh", "Snoo-48989", "OkFrame3668", "Personal_Slide_9421", "Hellmos_thorax", "0nce_A", "wadiostar", "initialpanik", "justarandomer_", "MoonlitShadow4416", "CHARILEwolf", "ImpeccableSloth33", "vulturesboy", "MathematicianFit1992", "Battlespot9999", "Subject_Low5199", "FriendlyStatus8165", "DryInvestment1906", "Creative-Store", "livvyloo94", "Reasonable_Eye_8072", "AcceptableSmoke9129", "MatikAX", "hypnoghoul", "i1xyz", "I_Forget_Stuff", "ASleepyB0i", "Careless-Scratch-658", "MelodiousDew", "AcediaIra", "Djurdjen", "Lou_Inc", "Ajackxe", "Guilty_Critic"]
             
mhdata = []

for username in usernames:
    try:
        user = reddit.redditor(username)
        
        _ = user.name
        
        submissions = user.submissions.new(limit=20)
        
        for submission in submissions:
            mhdata.append({
                "Username": username,
                "Post Text": submission.selftext
            })
    
    except NotFound:
        print(f"User {username} not found, skipping.")
    except Exception as e:
        print(f"An error occurred for user {username}: {e}")

mentalhealth = pd.DataFrame(mhdata)
mentalhealth['Diagnosis'] = "MH"
mentalhealth

## Lastly, with neither diagnosis

from prawcore.exceptions import NotFound

reddit = praw.Reddit(
    client_id="EgYhtuW9L0CD3LBIgvkvrw",
    client_secret="BeUvMfWktpPrGVQK5myrTaw8scgnLA",
    user_agent="Schiz/suicide correlation by /u/NecessaryClassic4135"
)

usernames = ["youre-welcome5557777", "maybesaydie", "Current_Variety_9577", "MileHighMontana", "Tasty_Ad_4082", "Solo-is-simpler", "-Moshe-", "spayne1111", "These_Quit_1692", "GendoIkari_82", "MistWeaver80", "AceCombat9519", "CaterpillarReady2709", "Asia_Correal", "Formal-Public-2652", "Certain-Ad-7578", "geekydreams", "Collt092", "winthropx", "DarkTwistedFantasee", "scotthallstandup", "Many-Ad-2154", "StabbesandSlash", "Sea_Swimmer_3179", "ben-costello", "CupcakeNatasha", "crackerzak", "Wide-Refuse-7815", "blimeyitsme", "fabry-sans", "Nym-ph", "Owen_D_Young", "Gamora3728", "Kyosji", "ae23ctm", "wjzwah3d", "kylorenismydad", "littleblackcar", "socalquest", "Chemical-Engineer979", "Subreddit-Guy", "Catmoonie", "KeyDonut2156", "DlGlTAL_IDENTlTY", "Large_Seesaw_232", "danman8001", "Real-Status-1414", "numbuhonestunnuh", "SignificantPea8021", "Natural-Brilliant-95"]
             
nonedata = []

for username in usernames:
    try:
        user = reddit.redditor(username)
        
        _ = user.name
        
        submissions = user.submissions.new(limit=20)
        
        for submission in submissions:
            nonedata.append({
                "Username": username,
                "Post Text": submission.selftext
            })
    
    except NotFound:
        print(f"User {username} not found, skipping.")
    except Exception as e:
        print(f"An error occurred for user {username}: {e}")

none = pd.DataFrame(nonedata)
none['Diagnosis'] = "Neither"
none

## Combining all of my data

import pandas as pd
all_posts = pd.concat([study, mentalhealth, none])
all_posts

all_posts.to_csv("all_posts.csv", index=False)


## Creating the training data set

reddit = praw.Reddit(
    client_id="EgYhtuW9L0CD3LBIgvkvrw",
    client_secret="BeUvMfWktpPrGVQK5myrTaw8scgnLA",
    user_agent="Schiz/suicide correlation by /u/NecessaryClassic4135"
)

subreddit = reddit.subreddit("SuicideWatch")

posts = []
for submission in subreddit.new(limit=None):
    title = submission.title
    text = submission.selftext
    combined_text = f"{title}. {text}"
    posts.append({
        "combined_text": combined_text,
        "is_suicidal": True  # Manually label as suicidal
    })

suicide = pd.DataFrame(posts)
suicide.to_csv("suicide_data.csv", index=False)

import random
reddit = praw.Reddit(
    client_id="EgYhtuW9L0CD3LBIgvkvrw",
    client_secret="BeUvMfWktpPrGVQK5myrTaw8scgnLA",
    user_agent="Schiz/suicide correlation by /u/NecessaryClassic4135"
)

subreddit_names = ["Coachella", "assassinscreed", "audiobooks", "aviation", "budgetfood"]

post_texts = []

for subreddit_name in subreddit_names:
    # get 4 random posts
    subreddit = reddit.subreddit(subreddit_name)
    random_submissions = random.sample(list(subreddit.new(limit=5)), 5)
    
    for submission in random_submissions:
        post_texts.append(submission.selftext)

nonsuicide = pd.DataFrame(post_texts)
nonsuicide.to_csv("nonsuicide_data.csv", index=False)

## I went through and took the top responses. I then self-identified whether they were suicidal text or not.

training_data = pd.read_csv('data_test.csv')
training_data

## Training + Testing Model!

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

X = training_data['compiled_text']
y = training_data['is_suicidal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

all_posts_vec = vectorizer.transform(all_posts['Post Text'])
all_posts['predicted_is_suicidal'] = model.predict(all_posts_vec)

print(all_posts)

## Analysis

import matplotlib.pyplot as plt

counts_df = all_posts.groupby(['Diagnosis', 'predicted_is_suicidal']).size().unstack(fill_value=0)
proportions_df = counts_df.div(counts_df.sum(axis=1), axis=0)
proportions_df.plot(kind='bar', stacked=True)

plt.xlabel('Diagnosis')
plt.ylabel('Proportion of Predictions')
plt.title('Proportion of True and False Predictions by Diagnosis')

plt.show()
proportions_df

grouped_df = all_posts.groupby(['Username', 'Diagnosis'])
at_least_one_suicidal = grouped_df['predicted_is_suicidal'].any().groupby('Diagnosis').sum()

total_users = grouped_df['Username'].nunique().groupby('Diagnosis').sum()

proportion_suicidal_users = at_least_one_suicidal / total_users

print(proportion_suicidal_users)

def count_suicide_mentions(posts):
    return sum(('suicide' in post) or ('suicidal' in post) for post in posts)

grouped = all_posts.groupby(['Username', 'Diagnosis'])['Post Text'].apply(count_suicide_mentions).reset_index(name='suicide_mentions')

users_with_suicide_mentions = grouped[grouped['suicide_mentions'] > 0]

users_with_suicide_mentions

# Did my model think this text from MistWeaver80 was suicidal?

try:
    all_posts[(all_posts.predicted_is_suicidal == True) and (all_posts.Username == "MistWeaver80")]
except Exception as e:
    print("No predictions of suicidal text from this user")


# It seems like this was not necessarily suicidal text...

all_posts[all_posts.Username == "MistWeaver80"].head(1)

from scipy.stats import chi2_contingency

contingency_table = pd.crosstab(all_posts['Diagnosis'] == 'SCH', all_posts['predicted_is_suicidal'])
print("Contingency Table:")
print(contingency_table)

chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"\nChi2 Statistic: {chi2}")
print(f"P-value: {p}")

if p < 0.05:
    print("There is a significant association between schizophrenia diagnosis and suicidal text (p < 0.05).")
else:
    print("There is no significant association between schizophrenia diagnosis and suicidal text (p >= 0.05).")

import numpy as np
from scipy.stats import chi2_contingency
from sklearn.linear_model import LinearRegression

new_posts = all_posts.copy()

new_posts['predicted_is_suicidal'] = new_posts['predicted_is_suicidal'].astype(int)

expected_categories = ['MH', 'SCH', 'Neither']

new_posts = pd.get_dummies(new_posts, columns=['Diagnosis'], prefix='Diagnosis')
for category in expected_categories:
    column_name = f'Diagnosis_{category}'
    if column_name not in new_posts.columns:
        new_posts[column_name] = 0


contingency_table = pd.crosstab(new_posts['Diagnosis_SCH'], new_posts['predicted_is_suicidal'])

chi2, p, dof, expected = chi2_contingency(contingency_table)

n = contingency_table.sum().sum()
cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
print(f"Cramér's V: {cramers_v}")

X = new_posts[['Diagnosis_MH', 'Diagnosis_Neither', 'Diagnosis_SCH']]
y = new_posts['predicted_is_suicidal']

model = LinearRegression()
model.fit(X, y)
r_squared = model.score(X, y)
print(f'R-squared: {r_squared}')
