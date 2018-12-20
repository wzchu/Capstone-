#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
#from ml_metrics import rmsle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


from mpl_toolkits.basemap import Basemap


# In[3]:


w01 = pd.read_csv('NYChour2017-1.csv')


# In[4]:


w01.head()


# In[5]:


w01["date"] = pd.to_datetime(w01["date"])
w01['month']=w01.date.dt.month 


# In[6]:


w001=w01.loc[(w01['month'] == 8)]
w001 = w001[['date']]


# In[7]:


w001.head()


# ## Create new01 with count 

# In[34]:


new01= new01.rename(columns={'starttime': 'date',})


# In[33]:


new01.head()


# In[35]:


mylist = new01['start station id'].unique()
stationdict = {elem : pd.DataFrame() for elem in mylist}
#d = {name: pd.DataFrame() for name in companies}
#companydict
for key in stationdict:
    stationdict[key] = new01[:][new01['start station id'] == key]


# In[36]:


wholelist = {}
for n in mylist:
   wholelist[n] = pd.DataFrame()


# In[37]:


for n in mylist:
  wholelist[n]=pd.merge(w001, stationdict[n], how='outer', on='date')
  wholelist[n]['start station id']= wholelist[n]['start station id'].fillna(value=n)
  wholelist[n]['count']= wholelist[n]['count'].fillna(value=0) 


# ## Import bike data 

# In[8]:


db8 = pd.read_csv('201708-citibike-tripdata.csv')


# In[9]:


db8.head()


# In[10]:


db11=db8[['start station id','start station latitude','start station longitude']]


# In[11]:


original=pd.DataFrame()


# In[12]:


original['start station id']=db11['start station id'].unique()


# In[13]:


b=pd.merge(original, db11,on='start station id',how='outer')


# In[14]:


original=b.drop_duplicates(subset=['start station id','start station latitude','start station longitude'], keep='first')


# In[15]:


mylist=db11['start station id'].unique()


# In[16]:


original.head()


# In[13]:


a=a.drop('index', axis=1)


# In[18]:


db= db8[['starttime','start station id']]
db["starttime"] = pd.to_datetime(db8["starttime"])


# In[31]:


db['starttime'] = db['starttime'].apply(lambda t: t.replace(second=0))
db['starttime'] = db['starttime'].apply(lambda t: t.replace(minute=0))
new01=db.groupby(['start station id','starttime']).size().reset_index(name='count')


# In[ ]:


## group weahter first 


# In[40]:


wholelist[72].head()
for n in mylist:
    wholelist[n]['year']=wholelist[n].date.dt.year 
    wholelist[n]['month']=wholelist[n].date.dt.month 
    wholelist[n]['day']=wholelist[n].date.dt.day
    wholelist[n]['hour']=wholelist[n].date.dt.hour
    wholelist[n]['weekday']=wholelist[n].date.dt.dayofweek


# In[21]:


new01['year']=new01.starttime.dt.year 
new01['month']=new01.starttime.dt.month 
new01['day']=new01.starttime.dt.day
new01['hour']=new01.starttime.dt.hour
new01['weekday']=new01.starttime.dt.dayofweek


# In[54]:


new01['day']=new01.date.dt.day


# In[159]:


#new01['ymddate']=pd.to_datetime(new01[['year', 'month', 'day']])


# In[163]:


#new01['weekday']=weekday(new01['ymddate'])


# In[30]:


new01


# ## This is station count  sort by number vulume 

# In[119]:


stationcount=new01.groupby(['start station id'])['count'].sum().reset_index(name='count')


# In[125]:


stationcount=stationcount.sort_values('count')


# In[127]:


for n in stationcount['start station id']:
    a=str(n)
    stationlist.append(a)
stationlist


# In[132]:


plt.figure(figsize=(20,10))

plt.bar(stationlist[-10:],stationcount['count'][-10:])


# ## This is station count  sort by WEEKAY vulume 
# 

# In[41]:


stationdict[519].head()


# In[42]:


month01519=stationdict[519].groupby(['weekday'])['count'].sum().reset_index(name='weekdaycount')


# In[43]:


month01519


# In[46]:


import matplotlib.ticker as ticker

plt.figure(figsize=(20,10))
ax = plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

plt.plot(month01519['weekday'],month01519['weekdaycount'])


# ## This is day count 

# In[76]:


month01=stationdict[519].groupby(['day'])['count'].sum().reset_index(name='daycount')


# In[134]:


month01


# In[143]:


import matplotlib.ticker as ticker

plt.figure(figsize=(20,10))
ax = plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

plt.plot(month01['day'],month01['daycount'])


# ## the weekday 

# In[169]:


stationdict[519]


# In[14]:


plt.figure(figsize=(20, 7))  
y=a['Start Station Latitude']
x=a['Start Station Longitude']
plt.ylim(40.500,40.850)
plt.xlim(-73.930,-74.0500)
plt.scatter(x,y, label='True Position')


# In[22]:


loc_df = pd.DataFrame()
loc_df['longitude'] = a['start station longitude']
loc_df['latitude'] = a['start station latitude']
print(loc_df)


# In[30]:


# print(loc_df)
#XY=loc_df.values.tolist()
# X=XY[:][1]
# print(X)
kmeans = KMeans(n_clusters=4,random_state=2,n_init = 10)
kmeans.fit(loc_df) 

plt.ylim(40.500,40.850)
plt.xlim(-73.930,-74.0500)

plt.scatter(loc_df['longitude'][0:],loc_df['latitude'][0:], c=kmeans.labels_, cmap='rainbow')  
plt.plot()


# In[32]:


loc_df['cl']=kmeans.labels_


# In[33]:


loc_df


# In[35]:


kmeans_1 = KMeans(n_clusters=4)
# Using fit_predict to cluster the dataset
#X = loc_df[['longitude','latitude']].values
predictions = kmeans_1.fit_predict(loc_df) 


# In[36]:


plt.ylim(40.500,40.850)
plt.xlim(-73.930,-74.0500)
plt.scatter(loc_df['longitude'][:],loc_df['latitude'][:], c=kmeans.labels_, cmap='rainbow')  
plt.plot()


# In[16]:


X = loc_df[['longitude','latitude']].values


# In[17]:


Ks = range(1, 10)
kmean = [KMeans(n_clusters=i).fit(X) for i in Ks]


# In[99]:


from scipy.spatial.distance import cdist, pdist

def plot_elbow(kmean, X):
    centroids = [k.cluster_centers_ for k in kmean]
    D_k = [cdist(X, center, 'euclidean') for center in centroids]
    dist = [np.min(D,axis=1) for D in D_k]

    # Total with-in sum of square
    wcss = [sum(d**2) for d in dist]
    tss = sum(pdist(X)**2)/X.shape[0]
    bss = tss-wcss

    plt.subplots(nrows=1, ncols=1, figsize=(8,8))
    ax = plt.subplot(1, 1, 1)
    ax.plot(Ks, bss/tss*100, 'b*-')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Percentage of variance explained (%)')
    plt.title('Elbow for KMeans clustering')
    plt.show()

plot_elbow(kmean, X)


# ##  Group the dataset to become 600*48 

# In[42]:


aa111=wholelist[519]


# In[44]:


aa111.head()


# In[103]:


aa111['day']=aa111['date'].dt.day


# In[60]:


stateiongroup = {elem : pd.DataFrame() for elem in mylist}


# In[61]:


for n in mylist: 
    pre=wholelist[n]
    pre['day']=pre['date'].dt.day
    aa107=pre.loc[(pre['day'] == 19)]
    aa119=pre.loc[(pre['day'] == 10)]
    new02=pd.concat([aa119,aa107])
    stateiongroup[n] = new02[:][new02['start station id'] == n]


# In[62]:


stateiongroup[72].head()


# In[63]:


for n in mylist:
    stateiongroup[n]=stateiongroup[n].drop('day',axis=1)
    stateiongroup[n]=stateiongroup[n].drop('year',axis=1)
    stateiongroup[n]=stateiongroup[n].drop('weekday',axis=1)
    stateiongroup[n]=stateiongroup[n].drop('month',axis=1)
    stateiongroup[n]=stateiongroup[n].drop('hour',axis=1)
    stateiongroup[n]=stateiongroup[n].drop('start station id',axis=1)
    #stateiongroup[n].insert(0, 'stationid',n)


# In[64]:


for n in mylist: 
    stateiongroup[n] = stateiongroup[n].set_index("date").transpose()
    stateiongroup[n].insert(0, 'stationid',n)


# In[65]:


stateiongroup[72]


# In[66]:


finalda=stateiongroup[72]


# In[67]:


for n in mylist[1:]:
    finalda=finalda.append(stateiongroup[n])


# In[68]:


finalda=finalda.set_index('stationid')


# In[70]:


finalda.to_csv('daily1019.csv', encoding='utf-8', index=True)


# # Clustering start 
# ### clustering the station with daily hour 

# In[219]:


finalda.head()


# In[199]:


clustring1 = pd.read_csv('daily1019.csv')


# In[221]:


x=clustring1.iloc[:,1:]


# In[76]:


Ks = range(1, 10)
kmean = [KMeans(n_clusters=i).fit(x) for i in Ks]


# In[77]:


from scipy.spatial.distance import cdist, pdist

def plot_elbow(kmean, X):
    centroids = [k.cluster_centers_ for k in kmean]
    D_k = [cdist(X, center, 'euclidean') for center in centroids]
    dist = [np.min(D,axis=1) for D in D_k]

    # Total with-in sum of square
    wcss = [sum(d**2) for d in dist]
    tss = sum(pdist(X)**2)/X.shape[0]
    bss = tss-wcss

    plt.subplots(nrows=1, ncols=1, figsize=(8,8))
    ax = plt.subplot(1, 1, 1)
    ax.plot(Ks, bss/tss*100, 'b*-')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Percentage of variance explained (%)')
    plt.title('Elbow for KMeans clustering')
    plt.show()

plot_elbow(kmean,x)


# In[ ]:


plt.ylim(40.500,40.850)
plt.xlim(-73.930,-74.0500)
plt.scatter(loc_df['longitude'][:],loc_df['latitude'][:], c=kmeans.labels_, cmap='rainbow')  
plt.plot()


# In[119]:


finalda1=finalda.reset_index()
#b=pd.merge(original, db11,on='start station id',how='outer')


# In[185]:


finalda.head()


# In[123]:


finalda1= finalda1.rename(columns={'stationid': 'start station id',})


# In[125]:


final=pd.merge(original, finalda1, how='outer', on='start station id')


# In[129]:


final1=final.set_index('stationid')


# In[222]:


kmeans = KMeans(n_clusters=4,random_state=2,n_init = 10)
kmeans.fit(x.values) 


# In[223]:


len(kmeans.labels_)


# In[177]:


original=original.drop('custering',axis=1)


# In[79]:


original


# In[80]:


mycluster1=original.copy()


# In[81]:


mycluster1=mycluster1.sort_values('start station id')


# In[224]:


finalda['cluster']=kmeans.labels_
####  myclustirng1 is for demand pattern using name clustering 


# In[210]:


plt.figure(figsize=(12,7))
plt.xlim(40.500,40.850)
plt.ylim(-73.920,-74.0500)
plt.scatter(mycluster1['start station latitude'][:],mycluster1['start station longitude'][:],c=mycluster1['custering'], cmap='rainbow')  


# ## Pattern For POI

# In[281]:


mycluster1


# In[90]:


codelist = pd.read_csv('codelist.csv')


# In[91]:


codelist=codelist.sort_values('station').reset_index(drop=True)


# In[93]:


codelist.head()


# In[94]:


aa=codelist.iloc[:,1:13]


# In[95]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
try1234 = scaler.fit_transform(aa)
#try1111 = scaler.fit_transform(aa)


# In[96]:


try1234=pd.DataFrame(try1234)


# In[97]:


try1234=try1234.drop(5,axis=1)


# In[98]:


try1234.insert(0, 'start station id',codelist['station'])


# In[99]:


try1234=try1234.set_index('start station id')


# In[101]:


try1234.head()


# In[156]:


kmeans = KMeans(n_clusters=6,random_state=2,n_init = 10)
kmeans.fit(try1234) 


# In[157]:


len(kmeans.labels_)


# In[159]:


try1234['cluster']=kmeans.labels_


# In[162]:


codelist['cluster']=kmeans.labels_


# In[61]:


try1111
try1111= try1111.rename(columns={'stationid': 'start station id ',})


# In[77]:


try1111.insert(0, 'start station id',codelist['station'])


# In[103]:


mycluster2=original.copy()


# In[104]:


mycluster2=mycluster2.sort_values('start station id').reset_index(drop=True)


# In[132]:


mycluster2['cluster']=kmeans.labels_


# In[257]:


mycluster2


# In[133]:


plt.figure(figsize=(12,7))
plt.xlim(40.500,40.850)
plt.ylim(-73.920,-74.0500)
plt.scatter(mycluster2['start station latitude'][:],mycluster2['start station longitude'][:],c=mycluster2['cluster'], cmap='rainbow')  


# In[263]:


Ks = range(1, 10)
kmean = [KMeans(n_clusters=i).fit(try1234) for i in Ks]
from scipy.spatial.distance import cdist, pdist

def plot_elbow(kmean, X):
    centroids = [k.cluster_centers_ for k in kmean]
    D_k = [cdist(X, center, 'euclidean') for center in centroids]
    dist = [np.min(D,axis=1) for D in D_k]

    # Total with-in sum of square
    wcss = [sum(d**2) for d in dist]
    tss = sum(pdist(X)**2)/X.shape[0]
    bss = tss-wcss

    plt.subplots(nrows=1, ncols=1, figsize=(8,8))
    ax = plt.subplot(1, 1, 1)
    ax.plot(Ks, bss/tss*100, 'b*-')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Percentage of variance explained (%)')
    plt.title('Elbow for KMeans clustering')
    plt.show()

plot_elbow(kmean,try1234)


# In[217]:


x=mycluster1['start station latitude'].loc[(mycluster1['custering'] ==2)].values
y=mycluster1['start station longitude'].loc[(mycluster1['custering'] ==2)].values


# In[218]:


plt.figure(figsize=(12,7))
plt.xlim(40.500,40.850)
plt.ylim(-73.920,-74.0500)
plt.scatter(x,y, cmap='rainbow')  


# In[264]:


from sklearn.metrics.cluster import adjusted_rand_score


# In[271]:


adjusted_rand_score(mycluster1['custering'],mycluster2['cluster'])  


# In[274]:


mycluster1=mycluster1.reset_index(drop=True)


# In[134]:


group10=mycluster1.loc[(mycluster1['custering'] ==0)]
group11=mycluster1.loc[(mycluster1['custering'] ==1)]
group12=mycluster1.loc[(mycluster1['custering'] ==2)]
group13=mycluster1.loc[(mycluster1['custering'] ==3)]


# In[229]:


group12


# In[135]:


group20=mycluster2.loc[(mycluster2['cluster'] ==0)]
group21=mycluster2.loc[(mycluster2['cluster'] ==1)]
group22=mycluster2.loc[(mycluster2['cluster'] ==2)]
group23=mycluster2.loc[(mycluster2['cluster'] ==3)]
group24=mycluster2.loc[(mycluster2['cluster'] ==4)]
group25=mycluster2.loc[(mycluster2['cluster'] ==5)]


# In[289]:


mycluster2['start station id'].loc[(mycluster2['cluster'] ==3)].values


# In[285]:


from math import*
  
def jaccard_similarity(x,y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)
  
print(jaccard_similarity(group10['start station id'].values,group20['start station id'].values))


# In[137]:


from math import*
  
def square_rooted(x):
  
   return round(sqrt(sum([a*a for a in x])),3)
  
def cosine_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return round(numerator/float(denominator),3)
  
print (cosine_similarity(group10['start station id'].values,group20['start station id'].values))


# In[138]:


for n in range(6):
    result=cosine_similarity(group10['start station id'].values,mycluster2['start station id'].loc[(mycluster2['cluster'] ==n)].values)
    print(result)


# In[140]:


for n in range(6):
    result=cosine_similarity(group11['start station id'].values,mycluster2['start station id'].loc[(mycluster2['cluster'] ==n)].values)
    print(result)


# In[142]:


for n in range(6):
    result=cosine_similarity(group12['start station id'].values,mycluster2['start station id'].loc[(mycluster2['cluster'] ==n)].values)
    print(result)


# In[143]:


for n in range(6):
    result=cosine_similarity(group13['start station id'].values,mycluster2['start station id'].loc[(mycluster2['cluster'] ==n)].values)
    print(result)


# In[130]:


clusterer = KMeans(n_clusters=4, random_state=10)
cluster_labels = clusterer.fit_predict(try1234)

silhouette_avg = silhouette_score(try1234, cluster_labels)
print("For n_clusters =",4,"The average silhouette_score is :", silhouette_avg)


# In[279]:


table=table.set_index("Cluster1")


# In[280]:


sns.heatmap(data=table,cmap="Set2",linecolor="white",linewidths=1,annot=True)


# In[256]:


table


# In[ ]:




