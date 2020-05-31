"""
Created on Thu Apr 30 13:20:45 2020
@author: DESHMUKH
RECOMMENDATION ENGINE
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import linear_kernel 
pd.set_option('display.max_columns',151)
pd.set_option('display.max_rows',200)

# ===================================================================================
# Business Problem :- Building a recommender system with the joke rating dataset using UBCF
# ===================================================================================

jokes = pd.read_excel("jokes ratings.xlsx",header = None)
jokes.rename(columns={0:'user_ID'},inplace=True)
jokes.head()
jokes.tail()
jokes.info()
jokes.isnull().sum()

# Replacing nan by 0
jokes = jokes.fillna(0)
jokes.tail()
jokes.isnull().sum()

# Replacing 99 by 0
jokes[jokes == 99] = 0

jokes = jokes.iloc[:1500,:] # i am slicing data into 1500 rows as large data not support in PC. 

# Removing first row
rating = jokes.iloc[:,1:] 

# Change the column indices from 0 to 150
rating.columns = range(rating.shape[1])

# Lets normalize all these ratings using StandardScaler and save them in ratings_diff variable
from sklearn.preprocessing import StandardScaler
ratings_diff = StandardScaler().fit_transform(rating)
ratings_diff

# Creating Cosine Matrix
from sklearn.metrics.pairwise import cosine_similarity
cosine = cosine_similarity(ratings_diff)
np.fill_diagonal(cosine, 0)

similarity_with_user = pd.DataFrame(cosine,index=jokes.user_ID)
similarity_with_user.columns=jokes.user_ID
similarity_with_user.head()


# Function to find top user
def find_n_neighbours(df,n):
    order = np.argsort(df.values, axis=1)[:, :n]
    df = df.apply(lambda x: pd.Series(x.sort_values(ascending=False)
           .iloc[:n].index, 
          index=['top{}'.format(i) for i in range(1, n+1)]), axis=1)
    return df


# Top 30 neighbours for each user
sim_user_30_m = find_n_neighbours(similarity_with_user,30)
sim_user_30_m.head()


                #-----------------------------------------#