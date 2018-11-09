
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
instances=np.matrix([[0,0],[0,1],[1,1],[1,0],[5,0]])
x=np.squeeze(np.asarray(instances[:,0]))
y=np.squeeze(np.asarray(instances[:,1]))
plt.cla()
plt.figure(1)
plt.scatter(x,y)
plt.show()


# In[3]:


k=2
distance='manhattan'
from sklearn.metrics import pairwise_distances
dist=pairwise_distances(instances,metric=distance)


# In[10]:


import heapq
from collections import defaultdict
k_distance=defaultdict(tuple)
for i in range(instances.shape[0]):
    distances=dist[i].tolist()
    ksmallest=heapq.nsmallest(k+1,distances)[1:][k-1]
    ksmallest_idx=distances.index(ksmallest)
    k_distance[i]=(ksmallest,ksmallest_idx)


# In[12]:


def all_indices(value,inlist):
    out_indices=[]
    idx=-1
    while True:
        try:
            idx=inlist.index(value,idx+1)
            out_indices.append(idx)
        except ValueError:
            break
    return out_indices
import heapq
k_distance_neig=defaultdict(list)
for i in range(instances.shape[0]):
    distances=dist[i].tolist()
    print("k distance neighbourhood",i)
    print(distances)
    ksmallest=heapq.nsmallest(k+1,distances)[1:]
    print(ksmallest)
    ksmallest_set=set(ksmallest)
    print(ksmallest_set)
    ksmallest_idx=[]


# In[24]:


local_reach_density=defaultdict(float)
for i in range(instances.shape[0]):
    no_neighbours=len(k_distance_neig[i])
    for neigh in k_distance_neig[i]:
        denom_sum+=max(k_distance[neigh[1]][0],neigh[0])
    local_reach_density[i] = no_neighbours/(1.0*denom_sum)


# In[21]:


lof_list=[]
for i in range(instances.shape[0]):
    lrd_sum=0
    rdist_sum=0
    for neigh in k_distance_neig[i]:
        lrd_sum+=local_reach_density[neigh[1]]
        rdist_sum+=max(k_distance[neigh[1]][0],neigh[0])
    lof_list.append((i,lrd_sum*rdist_sum))

