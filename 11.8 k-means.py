
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


def get_random_data():
    x_1=np.random.normal(loc=0.2,scale=0.2,size=(100,100))
    x_2=np.random.normal(loc=0.9,scale=0.1,size=(100,100))
    x=np.r_[x_1,x_2]
    return x


# In[4]:


x=get_random_data()

plt.cla()
plt.figure(1)
plt.title("Generated Data")
plt.scatter(x[:,0],x[:,1])
plt.show()


# In[5]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
def form_clusters(x,k):
    no_clusters=k
    model=KMeans(n_clusters=no_clusters,init="random")
    model.fit(x)
    labels=model.labels_
    print(labels)
    sh_score=silhouette_score(x,labels)
    return sh_score


# In[6]:


sh_scores=[]
for i in range(1,5):
    sh_score=form_clusters(x,i+1)
    sh_scores.append(sh_score)


# In[15]:


no_clusters =[i+1 for i in range(1,5)]

plt.figure(2)
plt.plot(no_clusters,sh_scores)
plt.title("Cluster Quality")
plt.xlabel("No of clusters k")
plt.ylabel("Silhouette Coefficient")
plt.show()

