# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:53:57 2018

@author: hp
"""

import matplotlib.pyplot as plt
from pylab import *
import xlrd
from xlrd import open_workbook
x_data=[]
y_data=[]
x_volte=[]
temp=[]
wb=open_workbook('my_data.xlsx')
for s in wb.sheets():
    print('Sheet:',s.name)
    for row in range(s.nrows):
        print('The row is :',row)
        values=[]
        for col in range(s.ncols):
            values.append(s.cell(row,col).value)
        print(values)
        x_data.append(values[0])
        y_data.append(values[1])
plt.plot(x_data,y_data,'bo-',label=u"Phase curve",linewidth=1)
plt.title(u"TR14 phase detector")
plt.legend()
ax=gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))




plt.xlabel(u"input-deg")
plt.ylabel(u"output-V")
plt.show()
print('over!')
