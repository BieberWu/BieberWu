
# coding: utf-8

# In[1]:


import numpy as np
a=np.array([1.1,2.2,3.3],dtype=np.float64)
a,a.dtype


# In[2]:


a.astype(int).dtype


# In[5]:


import numpy
numpy.array(object, dtype=None, copy=True, order=None, subok=False, ndmin=0)


# In[6]:


np.array([[[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]]])


# In[7]:


np.array([(1,2),(3,4),(5,6)])


# In[13]:


import numpy
numpy.arange(3, 7, 0.5, dtype='float32')


# In[14]:


np.linspace(0,10,10,endpoint=True)


# In[15]:


np.linspace(0,10,10,endpoint=False)


# In[16]:


np.ones((2,3))


# In[17]:


np.eye(5,4,3)


# In[18]:


np.fromfunction(lambda a, b:a+b,(5,4))


# In[21]:


b=np.array([[1,2,3],[4,5,6],[7,8,9]])
b


# In[22]:


b.T


# In[23]:


b.dtype


# In[24]:


b.imag


# In[25]:


b.real


# In[26]:


b.size


# In[27]:


b.itemsize


# In[28]:


b.nbytes


# In[29]:


b.ndim


# In[30]:


b.shape


# In[31]:


b.strides


# In[32]:


import numpy as np
np.arange(10).reshape((5,2))


# In[34]:


a=np.arange(10).reshape((2,5))
np.ravel(a)
np.ravel(a, order='F')


# In[35]:


a=np.ones((1,2,3))
np.moveaxis(a,0,-1)


# In[36]:


a=np.ones((1,4,3))
np.swapaxes(a,0,2)


# In[37]:


a=np.arange(4).reshape(2,2)
np.transpose(a)


# In[38]:


np.atleast_1d([1])
np.atleast_2d([1])
np.atleast_3d([1])


# In[39]:


a=np.arange(4).reshape(2,2)
np.asmatrix(a)


# In[41]:


a=np.array([[1,2],[3,4],[5,6]])
b=np.array([[7,8],[9,10]])
c=np.array([[11,12]])
np.concatenate((a,b,c),axis=0)


# In[42]:


a=np.array([[1,2],[3,4],[5,6]])
b=np.array([[7,8,9]])
np.concatenate((a,b.T),axis=1)


# In[43]:


a=np.array([1,2,3])
b=np.array([4,5,6])
np.stack((a,b))


# In[45]:


np.stack((a,b),axis=-1)


# In[46]:


a=np.arange(10)
np.split(a,5)


# In[47]:


a=np.arange(12).reshape(3,4)
np.delete(a,2,1)


# In[48]:


np.delete(a,2,0)


# In[50]:


a=np.arange(12).reshape(3,4)
b=np.arange(4)
np.insert(a,2,b,0)


# In[52]:


a = np.arange(6).reshape(2,3)
b = np.arange(3)
np.append(a,b)


# In[54]:


a=np.arange(10)
a.resize(2,5)
a


# In[55]:


a = np.arange(16).reshape(4,4)
np.fliplr(a)
np.flipud(a)


# In[56]:


np.random.rand(2,5)


# In[57]:


np.random.randn(1,10)


# In[58]:


np.random.randint(2,5,10)


# In[74]:


np.random.random_integers(2,5,10)


# In[73]:


np.random.random_integers(2,5,10)


# In[72]:


np.random.random_integers(2,5,10)


# In[62]:


np.random.random_sample([10])


# In[63]:


np.random.choice(10,5)


# In[64]:


import numpy as np
np.rad2deg(np.pi)


# In[65]:


a=np.random.randn(5)
a


# In[66]:


np.around(a)


# In[67]:


np.round_(a)


# In[69]:


np.rint(a)


# In[70]:


np.fix(a)


# In[71]:


np.floor(a)


# In[75]:


np.ceil(a)


# In[76]:


np.trunc(a)


# In[78]:


b=np.arange(10)
b


# In[80]:


np.prod(b)


# In[81]:


np.sum(b)


# In[82]:


np.nanprod(b)


# In[83]:


np.nansum(b)


# In[84]:


np.cumprod(b)


# In[85]:


np.cumsum(b)


# In[87]:


np.diff(b)


# In[88]:


a1=np.random.randint(0,10,5)
a2=np.random.randint(0,10,5)
a1,a2


# In[89]:


np.add(a1,a2)


# In[90]:


np.reciprocal(a1)


# In[91]:


np.negative(a1)


# In[92]:


np.multiply(a1,a2)


# In[93]:


np.divide(a1,a2)


# In[94]:


np.power(a1,a2)


# In[95]:


np.subtract(a1,a2)


# In[97]:


np.fmod(a1,a2)


# In[98]:


np.mod(a1,a2)


# In[99]:


np.modf(a1)


# In[100]:


np.remainder(a1,a2)


# In[101]:


import numpy as np
a=np.arange(10)
a


# In[102]:


a[1]


# In[103]:


a[[1,2,3]]


# In[105]:


b=np.arange(20).reshape(4,5)
b


# In[106]:


b[1,2]


# In[107]:


c=[[0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14],[15,16,17,18,19]]
c[1,2]


# In[108]:


c[1][2]


# In[109]:


d=np.arange(20).reshape(4,5)
d


# In[110]:


d[[1,2],[3,4]]


# In[111]:


e=np.arange(30).reshape(2,5,3)
e


# In[112]:


e[[0,1],[1,2],[1,2]]


# In[114]:


f=np.arange(10)
f


# In[115]:


f[:5]


# In[116]:


f[5:10]


# In[117]:


f[0:10:2]


# In[118]:


g=np.arange(20).reshape(4,5)
g


# In[119]:


g[0:3,2:4]


# In[120]:


g[:,::2]


# In[121]:


h=np.arange(10)
h


# In[122]:


h[1]=100
h


# In[123]:


numpy.sort(a,axis=-1,kind='quicksort',order=None)


# In[124]:


i=np.random.rand(20).reshape(4,5)
i


# In[125]:


i=np.random.randint(0,10,20)
i


# In[126]:


np.argmax(i)


# In[127]:


np.nanargmax(i)


# In[128]:


np.argmin(i)


# In[129]:


np.nanargmin(i)


# In[130]:


np.argwhere(i)


# In[131]:


np.nonzero(i)


# In[132]:


np.flatnonzero(i)


# In[133]:


np.count_nonzero(i)

