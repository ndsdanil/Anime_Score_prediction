import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
import re
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

#Checking data
train_data = pd.read_csv(r"C:\Users\MyPC\Documents\Python projects\Check google trends\archive\Anime.csv")
#we cant use the rows without information about anime raiting to train model, lets drop these rows with empty raiting
train_data = train_data.dropna(axis = 0, subset=['Rating'])
train_data = train_data.dropna(axis = 0, subset=['Release_year'])
chart_data = train_data.copy()
#lets remove the anime earlier 1980 
for index in train_data.index:
    if int(train_data.at[index,'Release_year']) < 2010:
        train_data = train_data.drop(index)
#Lets tell a bit story behind anime data

#Lets investigate How changed the anime rating since 1907 to 2022
#print(chart_data["Release_year"].max())
#print(chart_data["mean_rating_by_season"].unique())
plt.figure(figsize = (10,5))
plt.title("How changed the anime rating since 1907 to 2022")
sns.lineplot(x = "Release_year", y = "Rating", data = chart_data, color="red") # Your code here
#plt.figure(figsize = (10,5))
sns.scatterplot(x = "Release_year", y = "Rating", data = chart_data)
plt.show()

#Lets see on the quantity of released anime per year, its strange but every 6 years till 2021 we can see the spike of the anime quantity, why?
plt.figure(figsize = (20,5))
plt.title("How many anime were released since 1907 to 2022(anime-planet.com)")
plt.xticks([1920, 1940, 1960, 1980, 1985, 1991, 1997, 2003, 2009, 2015, 2020])
sns.histplot(chart_data["Release_year"])
plt.show()

#Lets prosid to investigate data, as we can see the most popular season to Release anime its Spring
plt.figure(figsize =(5,3))
plt.title("Overall anime quantity by season")
sns.histplot(chart_data["Release_season"] )
plt.show()

#Lets investigate Rating in relation to season, I figured mean Rating value for every season and depicted it in the plot. As we can see during the spring we have the highest ratings
chart_data["mean_rating_by_season"] = chart_data.groupby("Release_season")["Rating"].transform("mean")
print(chart_data["mean_rating_by_season"].unique())
plt.figure(figsize = (5,3))
plt.title("Mean anime rating by season for time period since 1907 to 2022")
sns.scatterplot(x = "Release_season", y = "mean_rating_by_season", data = chart_data) 
plt.show()

plt.figure(figsize = (15,10))
sns.scatterplot(y = "Episodes", x = "Rating", data = chart_data)
plt.show()

#in agreement with revealed data lets split columns into two categories: numeric columns and categorical columns
numeric_columns =['Episodes', 'Release_year', "Rating"]
categorical_columns = ['Type', 'Studio', 'Release_season', 'Related_Mange', 'Related_anime', 'Tags', 'Content_Warning' , 'staff'] #Can include later columns 'Tags, Content_Warning , Voice_actors (can analize in future the most frequent actors from anime with the high raiting), staff (use 'Original Creator', 'Director', 'Character Design', ': Music', )' further. 'Related_Mange', 'Related_anime' should be set as 1 and 0
final_columns = numeric_columns + categorical_columns
X_train = train_data[final_columns].copy()


#since we chosen columns, lets prepare them to fit the model, lets start from the Related_Mange and Related_anime and transform values in 1 or in 0 if there is no related manga or anime
X_train['Related_Mange'] = X_train['Related_Mange'].fillna(0)
X_train['Related_anime'] = X_train['Related_anime'].fillna(0)

for index in X_train.index:
    if X_train.at[index,'Related_Mange'] != 0:
        X_train.at[index,'Related_Mange'] = 1
    if X_train.at[index,'Related_anime'] != 0:
        X_train.at[index,'Related_anime'] = 1

#There are additional important columns which we can use: 'Tags, Content_Warning , staff (use 'Original Creator', 'Director', 'Character Design', ': Music', ) Lets investigate them
tags_list = []
content_warning_set = {}

for index in X_train.index:
    tags_list.append(str(X_train.at[index,'Tags']).split(',')) 
    tags_list1 = [item for sublist in tags_list for item in sublist]
    
tags_set = set(tags_list1)
        
#as you can see bellow there are a lot of values and we need reduce it, lets count how often these values repeat in dataframe, and make columns only for most frequent results.
tags_number = {}

for i in tags_list1:
    tags_number[i] = tags_list1.count(i)

        
#lets work with Content_Warning 
Content_Warning_list = []
for index in X_train.index:
    Content_Warning_list.append(str(X_train.at[index,'Content_Warning']).split(',,')) 
    Content_Warning_list1 = [item for sublist in Content_Warning_list for item in sublist]
Content_Warning_set =set(Content_Warning_list1)
for item in Content_Warning_set:
    item = item.strip()

#in tags_number variable I keep the value and number about how often this tag is used in dataset. Thus I can regulate the quantity of tags to use the most frequent values from dataset for further columns creation
tags_number_res = []
for key, value in tags_number.items():
    if value>200:
        tags_number_res.append(key.strip())
Content_Warning_res = []
for item in Content_Warning_set:
    Content_Warning_res.append(item.strip())
#print('Warning values')
#print(Content_Warning_res)

#we have some same values in tags_number_res and Content_Warning_res. Lets remove them from tags_number_res
for i in tags_number_res:
    if i in Content_Warning_res:
        tags_number_res.remove(i)
        
#print('Tags values after optimisation: ') 
#print(tags_number_res)

#Lets make a columns for staff, Tags and Content_Warning
staf_list = [': Original Creator', ': Director', ': Music']
for i in staf_list:
    X_train[i] = 'Null'
for i in tags_number_res:
    X_train[i] = 'Null'
for i in Content_Warning_res:
    X_train[i] = 'Null'
    
#lets work now with staff column. Wee need to fill the ': Original Creator',  '': Director'  ': Music' columns with respective values
for index in X_train.index:
    i = str(X_train.at[index,'staff'])
    if ': Original Creator' in i:
        x = re.search(r"(\w+\s\w+|\w+|\w.\w.\w.\w.|\w+.) : Original Creator", i).group(1)
        X_train.at[index, ': Original Creator'] = x
    if ': Chief Director' in i:
        x = re.search(r"(\w+\s\w+|\w+) : Chief Director", i).group(1)
        X_train.at[index, ': Director'] = x
    if ': Director' in i:
        x = re.search(r"(\w+\s\w+|\w+) : Director", i).group(1)
        X_train.at[index, ': Director'] = x
    if ': Music' in i:
        x = re.search(r"(\w+\s\w+|\w+|\w+\s\w+.|\w+.|\w.\w.\w.\w.|\w...) : Music", i).group(1)
        X_train.at[index, ': Music'] = x
        
#time to work with the tags_number_res columns
    b = str(X_train.at[index,'Tags'])
    for i in tags_number_res:
        if str(i) in b:
             X_train.at[index, i] = i

#time to work with the Content_Warning_res columns
    c = str(X_train.at[index,'Content_Warning'])
    for i in Content_Warning_res:
        if str(i) in c:
             X_train.at[index, i] = i
    try:
        v = X_train.at[index,'Release_year']
        X_train.at[index,'Release_year'] = int(v)
    except:
        n=0

#we need to round the Rating to decrease the viraety of target results
for index in X_train.index:  
    v = X_train.at[index,'Rating']
    X_train.at[index,'Rating'] = round(v, 0)
X_train['Rating'] = X_train['Rating'].astype('int64')
X_train['Episodes'] = X_train['Episodes'].fillna(0)

#now we have all required columns for futher encoding, so lets delete source columns: staff, Tags and Content_Warning
numeric_values_fin = ['Episodes', 'Release_year']
columns_for_labelencoder = ['Type','Studio', ': Original Creator', ': Director', ': Music', 'Release_season', 'Related_Mange', 'Related_anime','Action', 'Adventure', 'Fantasy', 'Shounen', 'Historical', 'Based on a Manga', 'Drama', 'Romance', 'Violence', 'Chinese Animation', 'Monsters','School Life', 'Supernatural', 'School Club', 'Original Work', 'Comedy','Superpowers', 'Based on a Light Novel', 'Non-Human Protagonists','Magic', 'Person in a Strange World', 'Music', 'Sci Fi', 'Parody','CG Animation', 'Seinen', 'Slice of Life', 'Recap', 'Ecchi','Based on a Video Game', 'Chibi', 'Short Episodes','Animal Protagonists', 'Anthropomorphic', 'Family Friendly', 'Mecha','Promotional', 'Shorts', 'Based on a Mobile Game', 'Vocaloid','Abstract', 'No Dialogue', 'Emotional Abuse', 'Drug Use','Physical Abuse', 'Animal Abuse', 'Suicide', 'Sexual Content','Explicit Sex', 'Mature Themes', 'Domestic Abuse', 'Nudity','Incest', 'Explicit Violence', 'Bullying', 'Sexual Abuse','Prostitution', 'Cannibalism', 'Self-Harm'] #'Studio', ': Original Creator', ': Director', ': Music',
#tags_and_warnings = ['Physical Abuse', 'Explicit Violence', 'Incest', 'Mature Themes', 'nan', 'Emotional Abuse', 'Sexual Content', 'Animal Abuse', 'Explicit Sex', 'Mature Themes', 'Physical Abuse', 'Bullying', 'Prostitution', 'Self-Harm', 'Bullying', 'Domestic Abuse', 'Emotional Abuse', 'Sexual Content', 'Violence', 'Explicit Violence', 'Suicide', 'Violence', 'Explicit Sex', 'Drug Use', 'Drug Use', 'Nudity', 'Incest', 'Nudity', 'Domestic Abuse', 'Sexual Abuse', 'Suicide', 'Cannibalism','Action', 'Adventure', 'Fantasy', 'Shounen', 'Demons', 'Historical', 'Swordplay', 'Based on a Manga', 'Drama', 'Romance', 'Shoujo', 'Contemporary Fantasy', '', 'Physical Abuse', 'Fantasy', 'Chinese Animation', 'Horror', 'Military', 'Monsters', 'School Life', 'Supernatural', 'Drama', 'War', 'Shounen', 'Sports', 'School Club', 'Tournaments', 'Original Work', 'Comedy', 'Superpowers', 'Based on a Light Novel', 'Non-Human Protagonists', 'Isekai', 'Magic', 'Person in a Strange World', 'RPG', 'Music', 'Mystery', 'Overpowered Main Characters', 'Sci Fi', 'Aliens', 'Feudal Japan', 'Gag', 'Parody', 'Urban Fantasy', 'Psychological', 'Comedy', 'CG Animation', 'Sci Fi', 'Based on a Visual Novel', 'Seinen', 'Superheroes', 'Cats', 'Episodic', 'Iyashikei', 'Slice of Life', 'Recap', 'Based on a Novel', 'Korean Animation', 'Adventure', 'Ecchi', 'Based on a Video Game', 'Robots', 'Chibi', 'Short Episodes', 'Animal Protagonists', 'Anthropomorphic', 'Family Friendly', 'Food and Beverage', 'Harem', 'Based on a 4-Koma Manga', 'Work Life', 'Mecha', 'Magical Girl', 'Henshin Heroes', 'Slice of Life', 'Promotional', 'Shorts', 'Based on a Mobile Game', 'Idols', 'Showbiz', 'Romance', 'Idols', 'Flash Animation', 'Vocaloid', 'Abstract', 'Vocaloid', 'No Dialogue', 'Animal Protagonists', 'Commercials']
columns_fin = numeric_values_fin + columns_for_labelencoder
y_train_fin = X_train.Rating
X_train_fin = X_train[columns_fin].copy()
#print(X_train_fin.info())

#print(X_train_fin.columns)
labelencoder = LabelEncoder()
for i in columns_for_labelencoder:
    X_train_fin[i] = labelencoder.fit_transform(X_train_fin[i])

y_train_fin= labelencoder.fit_transform(y_train_fin)

#lets assess the columns how userful they are for prediction of the target

discrete_features = X_train_fin.dtypes == int
def make_mi_scores(X_train_fin, y_train_fin, discrete_features):
    mi_scores = mutual_info_regression(X_train_fin, y_train_fin, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X_train_fin.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(X_train_fin,  y_train_fin, discrete_features)
print(mi_scores)  # show a few features with their MI scores

#Lets show on the plot the value of every column for the target
def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")

plt.figure(dpi=100, figsize=(20, 15))
plot_mi_scores(mi_scores)

model1 = xgb.XGBClassifier()  
scores = cross_val_score(model1, X_train_fin, y_train_fin,cv=5, scoring='f1_weighted')
print("MAE scores:\n", scores)
avg_score = np.mean(scores)
print(avg_score)
