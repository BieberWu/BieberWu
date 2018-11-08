
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
n_samples=100
fraction_of_outliers=0.1
number_inliers=int((1-fraction_of_outliers)*n_samples)
number_outliers=n_samples-number_inliers


# In[6]:


normal_data=np.random.randn(number_inliers,1)


# In[14]:


mean=np.mean(normal_data,axis=0)
std=np.std(normal_data,axis=0)
print ("Mean=(%0.2f) and Standard Deviation (%0.2f)"%(mean[0],std[0]))


# In[17]:


outlier_data=np.random.uniform(low=-9,high=9,size=(number_outliers,1))
total_data=np.r_[normal_data,outlier_data]
print("Size of input data=(%d,%d)"%(total_data.shape))
plt.cla()
plt.figure(1)
plt.title("Input points")
plt.scatter(range(len(total_data)),total_data,c='b')


# In[18]:


median=np.median(total_data)
b=1.4826
mad=b*np.median(np.abs(total_data-median))
outliers=[]
outlier_index=[]
print("Median Absolute Deviation=%.2f"%(mad))
lower_limit=median-(3*mad)
upper_limit=median+(3*mad)
print("Lower limit=%0.2f, Upper limit=%0.2f"%(lower_limit,upper_limit))
for i in range(len(total_data)):
    if total_data[i]>upper_limit or total_data[i]<lower_limit:
        print("Outlier %0.2f"%(total_data[i]))
        outliers.append(total_data[i])
        outlier_index.append(i)
plt.figure(2)
plt.title("Outliers using mad")
plt.scatter(range(len(total_data)),total_data,c='b')
plt.scatter(outlier_index,outliers,c='r')
plt.show()


# In[19]:


std=np.std(total_data)
mean=np.mean(total_data)
b=3
outliers=[]
outlier_index=[]
lower_limit=mean-b*std
upper_limit=mean+b*std
print("Lower limit=%0.2f,Upper limit=%0.2f"%(lower_limit,upper_limit))
for i in range(len(total_data)):
    x=total_data[i]
    if x>upper_limit or x<lower_limit:
        print("Outlier %0.2f"%(total_data[i]))
        outliers.append(total_data[i])
        outlier_index.append(i)
plt.figure(3)
plt.title("Outliers using std")
plt.scatter(range(len(total_data)),total_data,c='b')
plt.scatter(outlier_index,outliers,c='r')
plt.savefig("B04041 04 10.png")
plt.show()

