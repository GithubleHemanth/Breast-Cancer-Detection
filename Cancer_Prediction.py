#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r"C:\Users\HOME\Desktop\data.csv")


# In[3]:


df.head()


# In[4]:


df.columns


# In[5]:


df=df.drop('Unnamed: 32',axis=1)


# In[6]:


df.sample(10)


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df=df.drop("id",axis=1)


# In[10]:


df.head()


# In[11]:


df.diagnosis.value_counts()


# In[12]:


df.diagnosis.nunique()


# In[13]:


sample=212
df_malignant=df[df["diagnosis"]=="M"].sample(sample)


# In[14]:


df_malignant


# In[15]:


df_benign=df[df["diagnosis"]=="B"].sample(sample)


# In[16]:


df_benign.head()


# In[17]:


df_final=df_malignant.append(df_benign)


# In[18]:


df_final.diagnosis.value_counts()


# In[19]:


df_final.head()


# In[20]:


df_final.corr()


# In[21]:


sns.heatmap(data=df_final.corr(),annot=True,cmap="viridis")
plt.figure(figsize=(50,50))


# In[22]:


sns.countplot(data=df_final,x="diagnosis")


# In[23]:


sns.scatterplot(x="radius_mean",y="area_mean",hue="diagnosis",data=df)


# In[24]:


sns.boxplot(data=df_final,x="diagnosis",y="radius_mean")


# In[25]:


sns.violinplot(data=df_final,x="diagnosis",y="perimeter_mean")


# In[27]:


sns.displot(data=df_final,x="perimeter_mean",kde=True)


# In[28]:


sns.catplot(data=df_final,y="diagnosis",x="area_mean")


# In[30]:


sns.kdeplot(data=df_final,hue="diagnosis",x="perimeter_mean")


# In[34]:


sns.catplot(data=df_final,x="diagnosis",y="perimeter_mean",kind="bar")


# In[35]:


df_final.head()


# In[66]:


df_final["diagnosis"]=df_final["diagnosis"].apply(lambda x:1 if x=="M" else 0)


# In[67]:


X=df_final.iloc[:,1:].values
y=df_final.iloc[:,0].values


# In[68]:


X.shape


# In[69]:


y


# In[113]:


from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import cross_val_score


# In[103]:


ms=MinMaxScaler()
X=ms.fit_transform(X)



# In[104]:


X[0]


# In[105]:


y


# In[106]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[107]:


dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred=dt.predict(X_test)
accuracy=accuracy_score(y_pred,y_test)



# In[108]:


accuracy


# In[111]:


rf=RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred_rf=rf.predict(X_test)
acuracy_rf=accuracy_score(y_pred_rf,y_test)


# In[112]:


acuracy_rf


# In[120]:


#KFold cross validation
kfold=KFold(10)
results=cross_val_score(rf,X,y,cv=kfold)


# In[121]:


print(results)


# In[122]:


print(np.mean(results))


# In[123]:


#Stratified k fold cross validation
from sklearn.model_selection import StratifiedKFold
skfold=StratifiedKFold(n_splits=5)
results=cross_val_score(dt,X,y,cv=skfold)


# In[124]:


print(results)


# In[125]:


np.mean(results)


# In[126]:


from sklearn.model_selection import LeaveOneOut
loo=LeaveOneOut()
results=cross_val_score(rf,X,y,cv=loo)



# In[127]:


print(results)


# In[128]:


np.mean(results)


# In[ ]:




