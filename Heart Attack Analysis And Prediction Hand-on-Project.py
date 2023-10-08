#!/usr/bin/env python
# coding: utf-8
Project Content
Introduction
1.1 Examining the Project Topic
1.2 Recognizing Variables in Dataset
First Organization
2.1 Required Python Libraries
2.1.1 Basic Libraries
2.2 Loading the Dataset
2.3 Initial Analysis on Dataset
2.3.1 Analysis Output(1)
Preparation for Exploratory Data Analysis(EDA)
3.1 Examining the Missing Values
3.2 Examining the Unique Values
3.2.1 Analysis Output(2)
3.3 Seperating Variables(Numeric or Categorical)
3.4 Examining Statistics of Variables
Exploratory Data Analysis(EDA)
4.1 Uni-variate Analysis
4.1.1 Numerical Variables(Analysis with Distplot)
4.1.1.1 Analysis Output(4)
4.1.2 Categorical Variable(Analysis with Piechart)
4.1.2.2 Examining the Missing Data According to the Analysis Result
4.2 Bi-variate Analysis
4.2.1 Numeric Variables - Target Variables (Analysis with FacetGrid)
4.2.2 Categorical Variable - Target Variable (Analysis with Countplot)
4.2.3 Examining Numerical Variables Among Themselves (Analysis with Pair Plot)
4.2.4 Feature Scaling with Robust Scaler Method
4.2.5 Creating a New DataFrame with Melt() function
4.2.6 Numerical Varibales - Categorical Variables (Analysis with Swarmplot)
4.2.7 Numerical Variables - Categorical Variables (Analysis with Boxplot)
4.2.8 Relationship between variables (Analysis with Heatmap)
Preparation for Modelling
5.1 Dropping Columns with Low Correlation
5.2 Struggling Outliers
5.2.1 Visualizing Outliers
5.2.2 Dealing with Outliers
5.2.2.1 Trtbps Variable
5.2.2.2 Thalach Variable
5.2.2.3 Oldpeak
5.3 Determining Distribution of Numerical Variables
5.5 Applying One Hot Encoding Method to Categorical Variables
5.6 Feature Scaling with the Robust Scaler Method for Machine Learning Alogrithm
5.7 Seperating Data into Test and Training Set
Modelling
6.1 Logistic Regression
6.1.1 Cross Validation
# In[1]:


import numpy as np 
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns


# # Loading the Dataset

# In[2]:


df = pd.read_csv("heart.csv")


# In[3]:


df.head()


# #  Analysis Output(1) 

# In[4]:


df.head()


# In[5]:


new_columns = ["age","sex","cp","trtbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"]


# In[6]:


df.columns = new_columns


# In[7]:


df.head()


# In[8]:


print("Dataset Shape : ",df.shape)


# In[9]:


df.info()


# # Preparation for Exploratory Data Analysis(EDA)

# # Examining the Missing Values

# In[10]:


df.isnull()


# In[11]:


df.isnull().sum()


# In[12]:


isnull_number = []
for i in df.columns:
    x = df[i].isnull().sum()
    isnull_number.append(x)
pd.DataFrame(isnull_number,index = df.columns , columns = ["Total Missing Values"])


# In[13]:


pip install missingno


# In[14]:


#Visually show data by "missingno" library
import missingno
missingno.bar(df,color="b")


# # Examining the Unique Values 

# In[15]:


df.head()


# In[16]:


df["cp"].value_counts()


# In[17]:


df["cp"].value_counts().count()


# In[18]:


df["cp"].value_counts().sum()


# In[19]:


# Differentiate b/w Numerical and Categorical Values
unique_numbers = []
for i in df.columns:
    x=df[i].value_counts().count()
    unique_numbers.append(x)
pd.DataFrame(unique_numbers,index = df.columns,columns = ["Total Unique Numbers"])


# #  Analysis Output(2)

# #  Seperating Variables (Numeric or Categorical)

# In[20]:


numerical_var = ["age","trtbps","chol","thalach","oldpeak"]
categorical_var = ["sex","cp","fbs","restecg","exang","slope","ca","thal","target"]


# #  Examining the Statistics of Variables 

# In[21]:


#Apply on Numerical Variable
df[numerical_var].describe()


# In[22]:


sns.distplot(df["age"])


# In[23]:


sns.distplot(df["trtbps"],hist_kws = dict(linewidth = 1 , edgecolor = "k"));


# In[24]:


sns.distplot(df["trtbps"],hist_kws = dict(linewidth = 1 , edgecolor = "k"),bins=20)


# In[25]:


sns.distplot(df["chol"],hist= False)


# In[26]:


x , y = plt.subplots(figsize = (8,6))
sns.distplot(df["thalach"], hist = False , ax = y)
y.axvline(df["thalach"].mean(),color = "r", ls = "--")


# In[27]:


x , y = plt.subplots(figsize=(8,6))
sns.distplot(df["oldpeak"],hist_kws=dict(linewidth = 1,edgecolor="k"), bins = 20 , ax = y)
y.axvline(df["oldpeak"].mean(),color="r",ls="--")


# #  Uni-Variate Analysis 
#  Exploratory Data Analysis 
#  Numerical Variables(Analysis with Distplot) 

# In[28]:


numerical_var


# In[29]:


title_font = {"family":"arial","color":"darkred","weight":"bold","size":15}
axis_font = {'family':'arial' , 'color' : 'darkblue' , 'weight' : 'bold' , 'size' : 13}
for i in numerical_var:
    plt.figure(figsize=(8,6),dpi=80)
    sns.distplot(df[i],hist_kws=dict(linewidth=1 , edgecolor="k"), bins = 20)
    
    plt.title(i , fontdict = title_font)
    plt.xlabel(i , fontdict = axis_font)
    plt.ylabel("Density", fontdict=axis_font)
    
    plt.tight_layout()
    plt.show()


# In[30]:


numerical_axis_name = ["Age of the Patient" , "Resting Blood Pressure","Cholestrol","Maximum Heart Rate Achieved","ST Depression"]


# In[31]:


list(zip(numerical_var,numerical_axis_name))


# In[32]:


title_font = {'family':'arial','color':'darkred','weight':'bold','size':15}
axis_font = {'family':'arial','color':'darkblue','weight':'bold','size':13}
for i,z in list(zip(numerical_var,numerical_axis_name)):
    plt.figure(figsize = (8,6) , dpi = 80)
    sns.distplot(df[i], hist_kws = dict(linewidth=1,edgecolor="k"), bins = 20)
    
    plt.title(i , fontdict= title_font)
    plt.xlabel(z , fontdict= axis_font)
    plt.ylabel("Density" , fontdict= axis_font)
    
    plt.tight_layout()
    plt.show()


# #  Analysis Output(4) 
# 
# 4.1.2 Categorical Variables (Analysis with Piechart) 

# In[33]:


categorical_var


# In[34]:


categorical_axis_name = ['Gender' , 'Chest Pain Type' ,' Fasting Bloodpressure' , 'Resting Electrocardiographic Results' , 'Exercise Induced Angina' , 'Slope of ST Segment' , 'Number of Major Vessels ', 'Thal' , 'Target' ]


# In[35]:


list(zip(categorical_var,categorical_axis_name))


# In[36]:


df["cp"].value_counts()


# In[37]:


df["cp"].value_counts().sum()


# In[38]:


list(df["cp"].value_counts())


# In[39]:


list(df["cp"].value_counts().index)


# In[40]:


label_font = {'family':'arial' , 'color' : 'darkred' , 'weight' : 'bold' , 'size' : 15}
axis_font = {'family':'arial' , 'color' : 'darkblue' , 'weight' : 'bold' , 'size' : 13}

for i , z in list(zip(categorical_var,categorical_axis_name)):
    fig , ax = plt.subplots(figsize=(8,6))
    
    observation_values = list(df[i].value_counts().index)
    total_observation_values = list(df[i].value_counts())
    
    ax.pie(total_observation_values,labels=observation_values, autopct="%1.1f%%",startangle=110,labeldistance=1.1)
    ax.axis("equal")
    
    plt.title(i + "(" + z + ")")
    plt.legend()
    plt.show()


# # Examining the Missing Data According to the Analysis Result

# In[41]:


# 3 Obsn Values in "Thal" variabel , 4th variable is null(0) . Discarding missing values and replace them with meaningful Data
df[df["thal"]==0]


# In[42]:


df["thal"].replace(0,np.nan)


# In[43]:


df["thal"] = df["thal"].replace(0,np.nan)


# In[44]:


df.loc[[48,281],:]


# In[45]:


isnull_number = []
for i in df.columns:
    x=df[i].isnull().sum()
    isnull_number.append(x)
pd.DataFrame(isnull_number,index=df.columns,columns=["Total Missing Values"])


# In[46]:


# Filling Null Variable
df["thal"].fillna(2,inplace=True)


# In[47]:


df.loc[[48,281],:]


# In[48]:


df


# In[49]:


# Convert all values in Thal from float to integer type
df["thal"]= pd.to_numeric(df["thal"],downcast="integer")


# In[50]:


df.loc[[48,281],:]


# In[51]:


isnull_number = []
for i in df.columns:
    x = df[i].isnull().sum()
    isnull_number.append(x)
pd.DataFrame(isnull_number,index=df.columns,columns=["Total Missing Values"])


# In[52]:


# No missing values
df["thal"].value_counts()


# #  Bi-Variate Analysis 
# 
# 4.2.1 Numerical Variables - Target Variables ( Analysis with FacetGrid)

# In[53]:


numerical_var


# In[54]:


numerical_var.append("target")


# In[55]:


numerical_var


# In[56]:


label_font = {'family':'arial','color':'darkred','weight':'bold','size':15}
axis_font = {'family':'arial','color':'darkblue','weight':'bold','size':13}

for i , z in list(zip(numerical_var,numerical_axis_name)):
    graph = sns.FacetGrid(df[numerical_var],hue="target",height=5,xlim = ((df[i].min()-10) ,( df[i].max()+10)))
    graph.map(sns.kdeplot,i,shade=True)
    graph.add_legend()
    
    plt.title(i , fontdict=label_font)
    plt.xlabel(z , fontdict=axis_font)
    plt.ylabel("Density",fontdict=axis_font)
    plt.tight_layout()
    plt.show()


# In[57]:


df[numerical_var].corr()


# In[58]:


df[numerical_var].corr().iloc[:,[-1]]


# In[59]:


categorical_var


# In[60]:


label_font = {'family':'arial' , 'color':'darkred' , 'weight' : 'bold' , 'size' : 15}
axis_font = {'family':'arial' , 'color':'darkblue' , 'weight' : 'bold' , 'size' : 13}

for i ,z in list(zip(categorical_var,categorical_axis_name)):
    plt.figure(figsize=(8,5))
    sns.countplot(x= i , data = df[categorical_var], hue = "target")
    
    plt.title(i + "- target",fontdict=label_font)
    plt.xlabel(z,fontdict=axis_font)
    plt.ylabel("Target",fontdict=axis_font)
    
    plt.tight_layout()
    plt.show()


# In[61]:


df[categorical_var].corr()


# In[62]:


df[categorical_var].corr().iloc[:,[-1]]
# cp and exang has the most corelation (Moderate Correlation) . fbs has least corelation


# In[63]:


numerical_var


# In[64]:


numerical_var.remove("target")


# In[65]:


numerical_var


# In[66]:


graph = sns.pairplot(df[numerical_var],diag_kind="kde")
graph.map_lower(sns.kdeplot,levels=4,color=".2")
plt.show()


# # Feature Scaling with Robust Scaler Method

# In[67]:


from sklearn.preprocessing import RobustScaler


# In[68]:


robust_scaler = RobustScaler()


# In[69]:


scaled_data = robust_scaler.fit_transform(df[numerical_var])


# In[70]:


scaled_data


# In[71]:


df_scaled = pd.DataFrame(data=scaled_data,columns=numerical_var)


# In[72]:


df_scaled.head()
# Distance b/w Data are scaled to certain range by preserving their weights so structure of data is not destroyed


# # Creating a New DataFrame with Melt() function

# In[73]:


df_new = pd.concat([df_scaled,df.loc[:,"target"]],axis=1)


# In[74]:


df_new.head()


# In[75]:


melted_data = pd.melt(df_new ,id_vars="target",var_name="variables",value_name="value")


# In[76]:


melted_data.head()


# In[77]:


# Create prototype graphic by swarmplot
plt.figure(figsize=(8,5))
sns.swarmplot(x="variables",y="value",hue="target",data=melted_data)
plt.show()
# Variables with best correlations are thalach and oldpeak variables. At threshold of 0 , 30% misdetermination . Colors are mixed


# # Numerical Varibles - Categorical Variables (Analysis with swarmplot)

# # Categorical Variabels - Numerical Variables (Analysis with Boxplot)

# In[78]:


axis_font = {'family':'arial','color':'black','weight':'bold','size':13}

for i in df[categorical_var]:
    df_new = pd.concat([df_scaled,df.loc[:,i]],axis=1)
    
    melted_data = pd.melt(df_new,id_vars=i,var_name="variables",value_name="value")
    
    plt.figure(figsize=(8,5))
    sns.boxplot(data=melted_data,x="variables",y="value",hue=i)
    
    plt.xlabel("variables",fontdict=axis_font)
    plt.ylabel("value",fontdict=axis_font)
    
    plt.tight_layout()
    plt.show()
    


# # Relationship between variables (Analysis with Heatmap)

# In[79]:


df_scaled


# In[80]:


df_new2 = pd.concat([df_scaled,df[categorical_var]],axis=1)


# In[81]:


df_new2


# In[82]:


plt.figure(figsize=(15,10))
sns.heatmap(df_new2.corr(),annot=True,cmap="Spectral",linewidths=0.5)


# In[83]:


df_new2.corr()


# # Preparation for Modelling

# # Dropping Columns with Low Correlation

# In[84]:


df.head()


# In[85]:


df.drop(["fbs","restecg","chol"],axis=1,inplace=True)


# In[86]:


df.head()


# # Struggling Outliers
# Visualizing Outliers

# In[87]:


fig , (ax1 , ax2 , ax3 , ax4) = plt.subplots(1,4,figsize=(20,6))

ax1.boxplot(df["age"])
ax1.set_title("age")

ax2.boxplot(df["trtbps"])
ax2.set_title("trtbps")

ax3.boxplot(df["thalach"])
ax3.set_title("thalach")

ax4.boxplot(df["oldpeak"])
ax4.set_title("oldpeak")

plt.show()


# # Dealing with Outliers
#  Trtbps Variables

# In[88]:


from scipy import stats
from scipy.stats import zscore
from scipy.stats.mstats import winsorize


# In[89]:


z_scores_trtbps = zscore(df["trtbps"])
for threshold in range(1,4):
    print("Threshold Values : {}".format("threshold"))
    print("Number of Outliers : {}".format(len(np.where(z_scores_trtbps > threshold)[0])))
    print("------------------")


# In[90]:


df[z_scores_trtbps > 2][["trtbps"]]


# In[91]:


df[z_scores_trtbps > 2].trtbps.min()


# In[92]:


df[df["trtbps"]<170].trtbps.max()


# In[93]:


winsorize_percentile_trtbps = stats.percentileofscore(df["trtbps"],165) / 100


# In[94]:


print(winsorize_percentile_trtbps)


# In[95]:


1 - winsorize_percentile_trtbps


# In[96]:


trtbps_winsorize = winsorize(df.trtbps,(0,(1-winsorize_percentile_trtbps)))


# In[97]:


plt.boxplot(trtbps_winsorize)
plt.xlabel("trtbps_winsorize",color="b")
plt.show()


# In[98]:


df["trtbps_winsorize"] = trtbps_winsorize


# In[99]:


df.head()


# # Thalach Variable

# In[100]:


def iqr(df,var):
    q1 = np.quantile(df[var],0.25)
    q3 = np.quantile(df[var],0.75)
    diff = q3-q1
    lower_v = q1-(1.5*diff)
    upper_v = q3+(1.5*diff)
    return df[(df[var]<lower_v)|(df[var]>upper_v)]


# In[101]:


thalach_out = iqr(df,"thalach")


# In[102]:


thalach_out


# In[103]:


df.drop([272],axis=0,inplace=True)


# In[104]:


df["thalach"][270:275]


# In[105]:


plt.boxplot(df["thalach"])


# # Oldpeak Variable

# In[106]:


def iqr(df,var):
    q1 = np.quantile(df[var],0.25)
    q3 = np.quantile(df[var],0.75)
    diff = q3-q1
    lower_v =q1 - (1.5*diff)
    upper_v =q3 + (1.5*diff)
    return df[(df[var]<lower_v)|(df[var]>upper_v)]


# In[107]:


iqr(df,"oldpeak")


# In[108]:


# Donot remove the dataset and reduce the amount of Data. Apply winsorize method
df[df["oldpeak"]<4.2].oldpeak.max()


# In[109]:


winsorize_perrcentile_oldpeak = (stats.percentileofscore(df["oldpeak"],4)) / 100
print(winsorize_perrcentile_oldpeak)


# In[110]:


oldpeak_wonsorize = winsorize(df.oldpeak,(0,(1-winsorize_perrcentile_oldpeak)))


# In[111]:


df["oldpeak_winsorize"] = oldpeak_wonsorize


# In[112]:


df.head()


# In[113]:


df.drop(["trtbps","oldpeak"],axis=1,inplace=True)


# In[114]:


df.head()


# # Determining Distribution of Numerical Variables

# In[115]:


df.head()


# In[116]:


fig , (ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(11,5))

ax1.hist(df["age"])
ax1.set_title("age")

ax2.hist(df["trtbps_winsorize"])
ax2.set_title("trtbps_winsorize")

ax3.hist(df["thalach"])
ax3.set_title("thalach")

ax4.hist(df["oldpeak_winsorize"])
ax4.set_title("oldpeak_winsorize")

plt.show()


# In[117]:


# Find the skewedness of Data
df[["age","trtbps_winsorize","thalach","oldpeak_winsorize"]].agg(["skew"]).transpose()


# # Transformation Operations on Unsymmetrical Data

# In[118]:


df["oldpeak_winsorize_log"] = np.log(df["oldpeak_winsorize"])
df["oldpeak_winsorize_sqrt"] = np.sqrt(df["oldpeak_winsorize"])


# In[119]:


df.head()


# In[120]:


df[["oldpeak_winsorize","oldpeak_winsorize_log","oldpeak_winsorize_sqrt"]].agg(["skew"]).transpose()


# In[121]:


df.drop(["oldpeak_winsorize","oldpeak_winsorize_log"],axis=1,inplace=True)


# In[122]:


df.head()


# 
# # Applying One Hot Encoding Method to Categorical Variables

# In[123]:


df_copy = df.copy()


# In[124]:


df_copy.head()


# In[125]:


categorical_var


# In[126]:


categorical_var.remove("fbs")
categorical_var.remove("restecg")


# In[127]:


categorical_var


# In[128]:


df_copy = pd.get_dummies(df_copy,columns=categorical_var[:-1],drop_first=True)


# In[129]:


df_copy.head()


# # Feature Scaling with the Robust Scaler Method for Machine Learning Alogrithm

# In[130]:


numerical_var


# In[131]:


new_numeric_var = ["age","thalach","trtbps_winsorize","oldpeak_winsorize_sqrt"]


# In[132]:


robust_scaler = RobustScaler()


# In[133]:


df[new_numeric_var] = robust_scaler.fit_transform(df_copy[new_numeric_var])


# In[134]:


df_copy.head()


# # Seperating Data into Test and Training Set

# In[135]:


from sklearn.model_selection import train_test_split


# In[136]:


X = df_copy.drop(["target"],axis=1)
y = df_copy[["target"]]


# In[137]:


X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.1,random_state=3)


# In[138]:


X_train.head()


# In[139]:


y_train.head()


# In[140]:


print(f"X_train : {X_train.shape[0]}")
print(f"X_test : {X_test.shape[0]}")
print(f"y_train : {y_train.shape[0]}")
print(f"y_test : {y_test.shape[0]}")


# # Modelling
# Logistic Regression Alogrithm

# In[141]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[142]:


log_reg = LogisticRegression()
log_reg


# In[143]:


log_reg.fit(X_train,y_train)


# In[144]:


y_pred = log_reg.predict(X_test)


# In[145]:


y_pred


# In[146]:


accuracy = accuracy_score(y_test,y_pred)
print("Test Accuracy : {}".format(accuracy))


# # Cross Validation

# In[147]:


from sklearn.model_selection import cross_val_score


# In[148]:


scores=cross_val_score(log_reg,X_test,y_test,cv=10)


# In[149]:


print("Cross Validation Score : ",scores.mean())
#No overfitting. Model has not memorized

