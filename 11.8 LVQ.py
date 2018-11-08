
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
import numpy as np
from sklearn.metrics import euclidean_distances,classification_report
data=load_iris()
x=data['data']
y=data['target']
from sklearn.preprocessing import MinMaxScaler
minmax=MinMaxScaler()
x=minmax.fit_transform(x)


# In[7]:


R=2
n_classes=3
epsilon=0.9
epsilon_dec_factor=0.001


# In[54]:


class prototype(object):
    def _int_(self,class_id,p_vector,epsilon):
        self.class_id=class_id
        self.p_vector=p_vector
        self.epsilon=epsilon
    def update(self,u_vector,increment=True):
        if increment:
            self.p_vector=self.p_vector+self.epsilon*(u_vector-self.p_vector)
        else:
            self.p_vector=self.p_vector-self.epsilon*(u_vector-self.p_vector)
            def find_closest(in_vector,proto_vectors):
                closet=None
                closet_distance=99999
                for p_v in proto_vectors:
                    distance=euclidean_distances(in_vector.reshape(1,4),p_v.p_vector.reshape(1,4))
        if distance<closest_distance:
            closest_distance=distance
            closest=p_v
            return closest
    def find_class_id(test_vector,p_vectors):
        return find_closest(test_vector,p_vectors).class_id


# In[70]:


p_vectors=[]
for i in range(n_classes):
    y_subset=np.where(y==i)
    x_subset=x[y_subset]
    samples=np.random.randint(0,len(x_subset),R)
    for sample in samples:
        s=x_subset[sample]
        p=prototype(i,s,epsilon)
        p_vectors.append(p)
print("class id \t Initial protype vector\n")
for p_v in p_vectors:
  print(p_v.class_id,'\t',p_v.p_vector)
    
   
    


# In[35]:




