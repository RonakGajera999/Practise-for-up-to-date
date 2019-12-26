#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


df1=pd.read_csv("C:\\python csv file\melb_data.csv")
df1


# In[3]:


df1.head()


# In[4]:


df1.tail()


# In[5]:


df1.info()


# In[6]:


df1.describe()


# In[7]:


df1.drop(['Address','Method','Distance'],axis=1, inplace= True)


# In[8]:


df1


# In[9]:


df1.head()


# In[10]:


df1.drop(['Type','Rooms','CouncilArea','Regionname','Propertycount'],axis=1)


# In[11]:


df1.isnull().head()


# In[12]:


df1.fillna(0)


# In[13]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


sns.pairplot(df1)


# In[15]:


df1.plot(x="Landsize",y="Price",color='red',kind='line')
df1.plot(x="YearBuilt",y="Price",color='pink',kind='hist')
df1.plot(x="Bedroom2",y="Price",color='blue',kind='scatter')
plt.show()


# In[16]:


fig=plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(x='YearBuilt',y='Price',data=df1,color='red')
ax1.scatter(x='YearBuilt',y='Bedroom2',data=df1,color='blue')
plt.legend(loc='upper left');
plt.show()


# In[17]:


df1.shape


# In[18]:


sns.pairplot(df1,x_vars="Bedroom2", y_vars="Price",size=7,aspect=0.7,kind="scatter")


# In[19]:


x=df1["Price"]
x.head()


# In[20]:


y=df1['Bedroom2']
y.head()


# In[21]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=1000)


# In[22]:


print(type(x_train))
print(type(x_test))
print(type(y_train))
print(type(y_test))


# In[23]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[24]:


x_train=x_train[:, np.newaxis]
x_test=x_test[:, np.newaxis]


# In[25]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train, y_train)


# In[26]:


print(lr.intercept_)
print(lr.coef_)


# In[27]:


# y=mx+c (2.14+7.12-10^07*Bedroom2)


# # Prediction

# In[28]:


y_pred=lr.predict(x_test)


# In[29]:


type(y_pred)


# # computing RMSE and r^2 value

# In[30]:


import matplotlib.pyplot as plt
c=[i for i in range(1,4075,1)]
fig=plt.figure()
plt.plot(c,y_pred,color='blue',linewidth=1,linestyle='-')
plt.plot(c,y_test,color='red',linewidth=0.5,linestyle='-')
fig.suptitle("Actual and Predicted", fontsize=20)
plt.xlabel('Price', fontsize=18)
plt.ylabel("Badroom2", fontsize=20)


# In[31]:


c=[i for i in range(1,4075,1)]
fig=plt.figure()
plt.plot(c,y_test-y_pred,color='blue',linewidth=0.2,linestyle='--')
fig.suptitle("Errors", fontsize=20)
plt.xlabel("Index", fontsize=15)
plt.ylabel("y_test-y_pred", fontsize=15)


# In[32]:


from sklearn.metrics import mean_squared_error, r2_score
mse=mean_squared_error(y_test, y_pred)


# In[33]:


r_squared= r2_score(y_test,y_pred)


# In[34]:


print("mean_squared_error :"  ,mse)
print("r_squared_value :" , r_squared)


# # R_squared value is not good, can you try!!!!

# In[35]:


import matplotlib.pyplot as plt
plt.scatter(y_test,y_pred)
plt.xlabel("Test")
plt.ylabel("Predicted y")


# In[ ]:




