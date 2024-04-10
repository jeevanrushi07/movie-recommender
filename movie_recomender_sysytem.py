#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head(1)


# In[4]:


credits.head(1)


# In[5]:


movies = movies.merge(credits, on='title')


# In[6]:


movies.head(1)


# In[7]:


# genres
# id
# keywords
# title
# overview
# cast
# crew

movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[8]:


movies.head()


# In[9]:


movies.isnull().sum()


# In[10]:


movies.dropna(inplace=True)


# In[11]:


movies.duplicated().sum()


# In[12]:


movies.iloc[0].genres


# In[13]:


import ast
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[14]:


movies['genres']=movies['genres'].apply(convert)


# In[15]:


movies.head()


# In[16]:


movies['keywords']=movies['keywords'].apply(convert)


# In[17]:


movies.head()


# In[18]:


def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter !=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[19]:


movies['cast']=movies['cast'].apply(convert3)


# In[20]:


def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[21]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[22]:


movies.head()


# In[23]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[24]:


movies.head()


# In[25]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[26]:


movies.head()


# In[27]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[28]:


movies['tags']


# In[29]:


new_df = movies[['movie_id','title','tags']]


# In[30]:


new_df['tags']


# In[31]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[32]:


new_df['tags'][0]


# In[33]:


new_df['tags']=new_df['tags'].apply(lambda x: x.lower())


# In[34]:


new_df.head()


# In[35]:


pip install nltk


# In[36]:


import nltk


# In[37]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[38]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[39]:


new_df['tags'][0]


# In[40]:


new_df['tags'].apply(stem)


# In[41]:


from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features=5000,stop_words='english')


# In[42]:


vectors =cv.fit_transform(new_df['tags']).toarray()


# In[43]:


vectors


# In[44]:


cv.get_feature_names_out()


# In[45]:


ps.stem('dancing')


# In[46]:


stem('in the 22nd century, a parapleg marin is dispatch to the moon pandora on a uniqu mission, but becom torn between follow order and protect an alien civilization. action adventur fantasi sciencefict cultureclash futur spacewar spacecoloni societi spacetravel futurist romanc space alien tribe alienplanet cgi marin soldier battl loveaffair antiwar powerrel mindandsoul 3d samworthington zoesaldana sigourneyweav jamescameron')


# In[47]:


from sklearn.metrics.pairwise import cosine_similarity


# In[48]:


similarity=cosine_similarity(vectors)


# In[49]:


sorted(similarity[1])


# In[50]:


similarity


# In[51]:


sorted(list(enumerate(similarity[0])),reverse=True,key = lambda x:x[1])[1:6]


# In[52]:


new_df[new_df['title']=='Batman Begins'].index[0]


# In[53]:


new_df[new_df['title']=='Avatar'].index[0]


# In[54]:


def recommend(movie):
    movie_index = new_df[new_df['title']==movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
    return 


# In[56]:


recommend('Batman Begins')


# In[57]:


import pickle


# In[60]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[62]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




