
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
x=np.linspace(0,2*np.pi,100)
y=np.sin(x)
plt.plot(x,y)


# In[3]:


fig=plt.figure()
ax=fig.add_axes([0.1,0.1,0.8,0.8])
ax.plot(x,y)


# In[5]:


x=np.arange(0.0,5.0,0.02)
y=np.exp(-x)*np.cos(2*np.pi*x)
plt.plot(x,y)
plt.grid(color='gray')


# In[6]:


fig=plt.figure()
ax=fig.add_axes([0.1,0.1,0.8,0.8])
ax.grid(color='gray')
ax.plot(x,y)


# In[7]:


fig=plt.figure()
ax=fig.add_axes([0.1,0.1,0.8,0.8])
ax.grid(color='blue')
ax.plot(x,y)
ax.set_xlabel("x axis")
ax.set_xlim((-2,10))


# In[9]:


fig=plt.figure()
ax=fig.add_axes([0.1,0.1,0.8,0.8])
ax.grid(color='gray')
ax.set_xlabel("x axis")
ax.set_xlim((0,5))
ax.set_xticks(np.linspace(0,5,11))


# In[10]:


ax=plt.axes()
ax.plot(np.random.rand(50))
ax.yaxis.set_major_locator(plt.NullLocator())
ax.xaxis.set_major_formatter(plt.NullFormatter())


# In[11]:


from sklearn.datasets import fetch_olivetti_faces
faces=fetch_olivetti_faces().images
fig,ax=plt.subplots(5,5,figsize=(5,5))
fig.subplots_adjust(hspace=0,wspace=0)
for i in range(5):
    for j in range(5):
        ax[i,j].xaxis.set_major_locator(plt.NullLocator())
        ax[i,j].yaxis.set_major_locator(plt.NullLocator())
        ax[i,j].imshow(faces[10*i+j],cmap='bone')


# In[18]:


from matplotlib.ticker import MultipleLocator, FormatStrFormatter
t=np.linspace(0,100,100)
s=9.8*np.power(t,2)/2
fig,ax=plt.subplots(figsize=(8,4))
ax.plot(t,s)
ax.set_ylabel("displacement")
ax.set_xlim(0,100)
ax.set_xlabel('time')
xmajor_locator=MultipleLocator(20)
xmajor_formatter=FormatStrFormatter("%1.1f")
xminor_locator = MultipleLocator(5)
ax.xaxis.set_major_locator(xmajor_locator)
ax.xaxis.set_major_formatter(xmajor_formatter)
ax.xaxis.set_minor_locator(xminor_locator)
ymajor_locator = MultipleLocator(10000)
ymajor_formatter = FormatStrFormatter("%1.1f")
yminor_locator = MultipleLocator(5000)
ax.yaxis.set_major_locator(ymajor_locator)
ax.yaxis.set_major_formatter(ymajor_formatter)
ax.yaxis.set_minor_locator(yminor_locator)
ax.grid(True, which='major')
ax.grid(True, which='minor')
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(16)


# In[16]:


fig,ax=plt.subplots(3,3,sharex='col',sharey='row')


# In[20]:


fig,ax=plt.subplots(3,3,sharex='col',sharey='row')
for i in range(3):
    for j in range(3):
        ax[i,j].text(0.5,0.5,str((i,j)),fontsize=18,ha='center')


# In[21]:


for i in range(1,7):
    plt.subplot(2,3,i)
    plt.text(0.5,0.5,str((2,3,i)),fontsize=16,ha='center')


# In[22]:


fig=plt.figure()
ax1=fig.add_axes([0.1,0.1,0.8,0.8])
ax2=fig.add_axes([0.6,0.5,0.2,0.3])


# In[25]:


x=np.linspace(0,2*np.pi,100)
plt.plot(x,np.sin(x),linestyle="-.",color='blue')
plt.plot(x, np.cos(x), linestyle=":", color='red')


# In[26]:


plt.plot(range(10),linestyle='--',marker='o',markersize=16,markerfacecolor='b',color='r')
plt.grid(True)


# In[27]:


plt.plot(range(10),'-Dr',markersize=16,markerfacecolor='r',markevery=[2,4,6],linewidth=6)


# In[30]:


a=np.arange(0,3,0.02)
b=np.arange(0,3,0.02)
c=np.exp(a)
d=c[::-1]
line1,=plt.plot(a, c, 'k--', label="Model")  
line2=plt.plot(a, d, "r:", label='Data')[0]    
line3=plt.plot(a, c+d, 'b-', label="Total")
plt.legend(loc=0)    


# In[32]:


a = np.arange(0, 3, 0.02)
b = np.arange(0, 3, 0.02)
c = np.exp(a)
d = c[::-1]
line1, = plt.plot(a, c, 'k--', label="Model")   
line2 = plt.plot(a, d, "r:", label='Data')[0]    
line3 = plt.plot(a, c+d, 'b-', label="Total")
plt.legend((line1,line2),loc=0)


# In[36]:


import pandas as pd
cities = pd.read_csv("/Users/hp/Desktop/city_population.csv")
cities


# In[38]:


lat = cities['latd']
lon = cities['longd']
population = cities['population']
area = cities['area']
plt.scatter(lon, lat, label=None, 
            c=np.log10(population), 
            cmap='viridis', s=area, 
            linewidth=0, alpha=0.5)
plt.axis(aspect='equal')
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.colorbar(label="log$_{10}$(population)")    
for area in [100, 300, 500]:
    plt.scatter([], [], c='k', alpha=0.3, s=area, label=str(area) + ' km$^2$')
plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='City Area')
plt.title("Jiangsu Cities: Area and Population")


# In[39]:


x=np.linspace(-np.pi,np.pi,9)
plt.plot(x,np.cos(x),"Dr",markerfacecolor='b',markersize=12)
plt.grid(True)


# In[40]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=[2,10,4,8,6]
position=[1,2,3,4,5]
plt.bar(x=position,height=data)


# In[42]:


position=[1,2.5,3,4.5,5]
plt.bar(x=position,height=data,width=0.4,bottom=[3,0,5,0,1])
plt.grid(1)


# In[44]:


data=[2,10,4,8,6]
position=[1,2,3,4,5]
labels=["Beijing","Soochow","Shanghai","Hangzhou","Hongkong"]
plt.bar(x=position,height=data,width=0.4,color='b',edgecolor='r',linestyle='--',linewidth=3,hatch='x',tick_label=labels)


# In[47]:


positon=np.arange(1,6)
a=np.random.random(5)
b=np.random.random(5)
plt.bar(position,a,label='a',color='b')
plt.bar(position,b,bottom=a,label='b',color='r')
plt.legend(loc=0)


# In[53]:


position = np.arange(1, 6)
a = np.random.random(5)
b = np.random.random(5)
total_width = 0.8
n = 2
width = total_width / n
position = position - (total_width - width) / n
plt.bar(position, a, width=width, label='a', color='b')
plt.bar(position + width, b, width=width, label='b', color='r')
plt.legend(loc=0)


# In[54]:


position=np.arange(1,6)
a=np.random.random(5)
plt.barh(position,a)


# In[56]:


position = np.arange(1, 6)
a = np.random.random(5)
b = np.random.random(5)
plt.barh(position,a,color='g',label='a')
plt.barh(position, -b, color='r', label='b')
plt.legend(loc=0)


# In[57]:


fig,ax=plt.subplots(1,2)
data=[1,5,9,2]
ax[0].boxplot([data])
ax[0].grid(True)
ax[1].boxplot([data],showmeans=True)
ax[1].grid(1)


# In[58]:


np.random.seed(12345)
data=pd.DataFrame(np.random.rand(5,4),columns=['A','B','C','D'])
data.boxplot(sym='r*',vert=False,meanline=False,showmeans=True)


# In[59]:


x=[2,4,6,8]
fig,ax=plt.subplots()
labels=['A','B','C','D']
colors=['red','yellow','blue','green']
explode=(0,0.1,0,0)
ax.pie(x,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True,startangle=90,radius=1.2)
ax.set(aspect='equal',title='Pie')


# In[60]:


fig=plt.figure()
mu=100
sigma=15
x=mu+sigma*np.random.randn(10000)
num_bins=50
n,bins,patches=plt.hist(x,num_bins,density=True,facecolor='blue',alpha=0.5,color='r')


# In[61]:


import matplotlib.mlab as mlab
fig = plt.figure()
mu = 100    
sigma = 15   
x = mu + sigma * np.random.randn(10000)
num_bins = 50
n, bins, patches = plt.hist(x, num_bins, density=True, facecolor='blue', alpha=0.5, color='r')  
y=mlab.normpdf(bins,mu,sigma)
plt.plot(bins,y,'r--')


# In[62]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
fig=plt.figure()
ax=plt.axes(projection='3d')
x_line=np.linspace(0,15,1000)
y_line=np.sin(x_line)
z_line=np.cos(x_line)
ax.plot3D(x_line,y_line,z_line,"b")
x_point=15*np.random.random(100)
y_point=np.sin(x_point)+0.1*np.random.randn(100)
z_point=np.cos(x_point)+0.1*np.random.randn(100)
ax.scatter3D(x_point,y_point,z_point,c=x_point,cmap="Reds")


# In[63]:


u=np.linspace(0,2*np.pi,30)
v=np.linspace(-0.5,0.5,8)/2.0
v,u=np.meshgrid(v,u)
phi=0.5*u
r = 1 + v * np.cos(phi)
x = np.ravel(r * np.cos(u))    
y = np.ravel(r * np.sin(u))
z = np.ravel(v * np.sin(phi))
from matplotlib.tri import Triangulation
tri=Triangulation(np.ravel(v),np.ravel(u))
ax=plt.axes(projection='3d')
ax.plot_trisurf(x,y,z,triangles=tri.triangles,cmap='viridis',linewidths=0.2)
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)


# In[64]:


import seaborn as sns
iris=sns.load_dataset("iris")
iris.head()


# In[65]:


sns.swarmplot(x="species",y="petal_length",data=iris)


# In[67]:


titanic=sns.load_dataset("titanic")
titanic.head()


# In[68]:


g=sns.barplot(x="class",y="survived",hue="sex",data=titanic)


# In[69]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data=np.random.multivariate_normal([0,0],[[5,2],[2,2]],size=2000)
data=pd.DataFrame(data,columns=['x','y'])
for col in "xy":
    plt.hist(data[col],density=True,alpha=0.5)


# In[73]:


data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
data = pd.DataFrame(data, columns=['x', 'y'])

for col in "xy":
     sns.kdeplot(data[col], shade=True)


# In[74]:


sns.distplot(data['x'])
sns.distplot(data['y'])


# In[75]:


sns.kdeplot(data)


# In[76]:


with sns.axes_style("white"):
    sns.jointplot('x','y',data,kind='kde')


# In[77]:


with sns.axes_style("white"):
    sns.jointplot('x','y',data,kind='hex')


# In[94]:


iris = sns.load_dataset("iris")
sns.pairplot(iris, hue="species",height=2.5)


# In[124]:


tips=sns.load_dataset("tips")
tips['tip_pct']=100*tips['tip']/tips['total_bill']
tips


# In[126]:


grid=sns.FacetGrid(tips,row='sex',col='time',margin_titles=True)
grid.map(plt.hist,"tip_pct",bins=np.linspace(0,40,15))
get_ipython().run_line_magic('pinfo', 'plt.hist')


# In[125]:


g=sns.FacetGrid(tips,col="time",row="smoker")
g.map(plt.scatter,"total_bill","tip",edgecolor="w")
#解析见下图


# In[92]:


g = sns.FacetGrid(tips, col="time",  hue="smoker")
g.map(plt.scatter, "total_bill", "tip", edgecolor="w").add_legend()
#​根据用餐者是否抽烟，及其就餐时间划分为4类，横轴为总餐费，纵轴为小费金额，由图可知，尤其是在晚餐时段，抽烟的消费者支付的小费比例离散性大于不抽烟的消费者


# In[93]:


bins=np.arange(0,65,5)
g=sns.


# In[95]:


iris = sns.load_dataset("iris")


# In[102]:


sns.pairplot(iris,hue="species",height=2.5)


# In[101]:


bins = np.arange(0, 65, 5)
g = sns.FacetGrid(tips, col="day", height=4, aspect=.5)
g.map(plt.hist, "total_bill", bins=bins)


# In[103]:


g=sns.FacetGrid(tips,col='smoker',col_order=["Yes","No"])
g.map(plt.hist,"total_bill",bins=bins,color='m')
#有图可知，不抽烟的消费者明显多于抽烟的消费者，此外，不抽烟的消费者在20元左右的区间的比重大于抽烟的消费者


# In[104]:


kws=dict(s=50,linewidth=.5,edgecolor="w")
g=sns.FacetGrid(tips,col="sex",hue="time",palette="Set1",hue_order=["Dinner","Lunch"])
g.map(plt.scatter,"total_bill","tip",**kws).add_legend()
#男性晚餐消费金融总体来说大于午餐消费金额，小费比例大体相同；而午餐或晚餐，以及其消费比例对女性而言变化均不大


# In[109]:


pal=dict(Lunch="seagreen",Dinner="gray")
g=sns.FacetGrid(tips,col="sex",hue="time",palette=pal,hue_order=["Dinner","Lunch"])
g.map(plt.scatter,"total_bill","tip",**kws).add_legend()


# In[112]:


g=sns.FacetGrid(tips,col="sex",hue="time",palette=pal,hue_order=["Dinner","Lunch"],hue_kws=dict(marker=["^","v"]))
g.map(plt.scatter,"total_bill","tip",**kws).add_legend()


# In[114]:


att=sns.load_dataset("attention")
g=sns.FacetGrid(att,col="subject",col_wrap=5,height=1.5)
g.map(plt.plot,"solutions","score",marker=".")


# In[115]:


from scipy import stats
def qplot(x,y,**kwargs):
    _,xr=stats.probplot(x,fit=False)
    _,yr=stats.probplot(y,fit=False)
    plt.scatter(xr,yr,**kwargs)
g=sns.FacetGrid(tips,col="smoker",hue="sex")
g.map(qplot,"total_bill","tip",**kws).add_legend()


# In[117]:


with sns.axes_style('white'):
    sns.catplot("total_bill","tip",data=tips,kind="box")
    g.set_axis_labels('Day','Today Bill')


# In[119]:


with sns.axes_style('white'):
    sns.jointplot("total_bill","tip",data=tips,kind="hex")


# In[120]:


sns.jointplot("total_bill","tip",data=tips,kind="reg")

