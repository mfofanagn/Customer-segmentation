import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import os
import sweetviz
from sklearn import preprocessing
import scipy.io.wavfile as wavfile
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import chi2_contingency
from  scipy.stats import  boxcox

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

## Data loading and exploratory data analysis

#os.chdir("D:/FOAD/DSTI/Data Glacer/Projects/final/")
#os.chdir("D:/DSTI/Data Glacer/Projects/final/")
os.chdir("D:/Data Glacer/Projects/final/")
customer = pd.read_csv("cust_seg.csv", low_memory=False)

#Print dataset structure
print(customer.info())

# Missing values checking

# Total missing value
print(customer.isnull().values.sum())

# Missing value as per column
print(customer.isnull().sum(axis=0))

# Initial Correlation matrix

#correlations = customer.corr()
#correlations.to_csv("corr_Init.csv")

#sns.heatmap(correlations)
#plt.show()

# Initial Automated Explotaory Data Analysis
#report = sweetviz.analyze(customer,pairwise_analysis='on')
#report.show_html(open_browser=True, filepath="D:/FOAD/DSTI/Data Glacer/Projects/final/EDA_Cust_Initial.html", layout='widescreen')


## Data Preprocessing

#Dropping numbering columns
customer = customer.drop('Unnamed: 0',1)

#Dropping column with unique categorical value
customer = customer.drop('tipodom',1)

#Dropping columns with low entropy
customer = customer.drop('ind_nuevo',1)
customer = customer.drop('indrel',1)
customer = customer.drop('indrel_1mes',1)
customer = customer.drop('indresi',1)
customer = customer.drop('indfall',1)
customer = customer.drop('ind_ahor_fin_ult1',1)
customer = customer.drop('ind_aval_fin_ult1',1)
customer = customer.drop('ind_cder_fin_ult1',1)
customer = customer.drop('ind_deco_fin_ult1',1)
customer = customer.drop('ind_deme_fin_ult1',1)
customer = customer.drop('ind_pres_fin_ult1',1)
customer = customer.drop('ind_viv_fin_ult1',1)

customer = customer.drop('ind_hip_fin_ult1',1)
customer = customer.drop('ind_plan_fin_ult1',1)
customer = customer.drop('ind_valo_fin_ult1',1)
customer = customer.drop('ind_fond_fin_ult1',1)
customer = customer.drop('ind_ctma_fin_ult1',1)
customer = customer.drop('ind_ctju_fin_ult1',1)
customer = customer.drop('indext',1)

customer = customer.drop('ind_empleado',1)
customer = customer.drop('pais_residencia',1)

#Dropping columns with high percentage of missing values
customer = customer.drop('ult_fec_cli_1t',1)
customer = customer.drop('conyuemp',1)

#Dropping high correlated columns  (cor with ind_nom_pens_ult1 >0.8 )
customer = customer.drop('ind_cno_fin_ult1',1)
customer = customer.drop('ind_nomina_ult1',1)
customer = customer.drop('cod_prov',1)

#Dropping columns with high cardinality
customer = customer.drop('ncodpers',1)
customer = customer.drop('fecha_alta',1)  # Garder peut etre ce champ en l'encodant ???

#NA dropping
#customer2 = customer.dropna(subset=['sexo', 'fecha_alta'])


#Data conversion
customer.age =  pd.to_numeric(customer.age, errors="coerce")
customer.antiguedad =  pd.to_numeric(customer.antiguedad, errors="coerce")

#customer.fecha_dato =  pd.Categorical(customer.fecha_dato)

customer['ind_actividad_cliente'] = customer['ind_actividad_cliente'].fillna(9)
customer['ind_actividad_cliente'] = customer['ind_actividad_cliente'].astype(int)

customer['ind_nom_pens_ult1'] = customer['ind_nom_pens_ult1'].fillna(9)
customer['ind_nom_pens_ult1'] = customer['ind_nom_pens_ult1'].astype(int)

#Outlear dropping
#customer3 = customer.drop(customer[customer.antiguedad < 0].index)

# Manage missing value and outlear in numerical variable
#To manage missing data, we replace
#numerical variable missing value by the mean of the variable
#categorical varaible missing value  by
#   9 for one digit variable type
#   U for one character variable type
#   Unknown for textual varible  type
#
#For outlear in field 'antiguedad', we replace outlear value with the median value

customer['renta'] = customer['renta'].fillna((customer['renta'].mean()))
customer['age'] = customer['age'].fillna((customer['age'].mean()))
customer['antiguedad'] = customer['antiguedad'].fillna((customer['antiguedad'].median()))
customer.loc[customer['antiguedad'] < 0, 'antiguedad'] = customer['antiguedad'].median()

# Manage missing value in categorical variable
customer['sexo'] = customer['sexo'].fillna("U")
customer['canal_entrada'] = customer['canal_entrada'].fillna("Unknown")

customer['nomprov'] = customer['nomprov'].fillna("Unknown")
customer['ind_nom_pens_ult1'] = customer['ind_nom_pens_ult1'].fillna(9)

customer['tiprel_1mes'] = customer['tiprel_1mes'].fillna("U")
#customer['ind_actividad_cliente'] = customer['ind_actividad_cliente'].fillna(9)

"""
# Boxplot Box Plot of renta by ind_dela_fin_ult1
fig = plt.figure(figsize= (10, 10))
ax = fig.add_subplot(111)

ax.set_title("Box Plot of antiguedad", fontsize= 20)
ax.set
data = customer['antiguedad']

ax.boxplot(data,showmeans= True)

plt.xlabel("")
plt.ylabel("Antiquedad")
plt.show()
"""

# Data scaling and gaussian transformation  for numerical variables

sc = StandardScaler()

customer.loc[:,['age']] = np.log(customer.loc[:,['age']])
customer.loc[:,['age']] = sc.fit_transform(customer.loc[:,['age']])

customer.loc[:, ['antiguedad']] = np.log(customer.loc[:,['antiguedad']]+1)
customer.loc[:,['antiguedad']] = sc.fit_transform(customer.loc[:,['antiguedad']])

customer.loc[:, ['renta']] = np.log(customer.loc[:,['renta']])
customer.loc[:, ['renta']] = sc.fit_transform(customer.loc[:, ['renta']])


#Dropping duplicated rows after some preprocessing
customer2 = customer.drop_duplicates()

# Fix index after row update
customer2 = customer2.reset_index(drop=True)

# Data structure after preprocessing
print(customer2.info())

# Missing value checking as per column after preprocessing
print(customer2.isnull().sum(axis=0))


# Correlation analysis

# Numerical relationship between all numerical features

correlations = customer2.corr()
#correlations.to_csv("corr_Fin_NoScaler.csv")
correlations.to_csv("corr_Fin.csv")
sns.heatmap(correlations)
plt.show()


# Categorical relationship between two categorical features
# Chi-Square Test for independance between 'tiprel_1mes' and 'ind_actividad_cliente'
# Null hypothesis : features  tiprel_1mes and ind_actividad_cliente are independant

obs = customer2[['tiprel_1mes', 'ind_actividad_cliente']]
contingency = pd.crosstab(obs['tiprel_1mes'], obs['ind_actividad_cliente'])
print(contingency)
c, p, dof, expected = chi2_contingency(contingency)
print("Chisq pValue  {0}".format(p))
print("Chisq dof     {0}".format(dof))

"""
pValue equal 0 which means that we can reject null hypothesis : 
So tiprel_1mes and ind_actividad_cliente are strongly dependant
"""

# Categorical and numerical relationship

# Anova test

# Anova test between  feature 'renta' and 'ind_dela_fin_ult1'
# We need to test if it exists relationship between long term deposit and Gross income of the household
# The null hypothesis is the mean of gross income between category of long term deposit is equal
import scipy.stats as stats

f1, p1 = stats.f_oneway(customer2['renta'][customer2['ind_dela_fin_ult1'] == 1],
                        customer2['renta'][customer2['ind_dela_fin_ult1'] == 0])

print("Anova 1 pValue  {0}".format(p1))
print("Anova 1 statistic  {0}".format(f1))

"""
pValue equal 0 which means that we can reject null hypothesis : 
So, long term deposit could be related to gross income 
We can observe this fact also in the Boxplot
"""

# Anova test between  feature 'age' and 'ind_dela_fin_ult1'
# We need to test if it exists relationship between long term deposit and age
# The null hypothesis is the mean of age  between category of long term deposit is equal
f2, p2 = stats.f_oneway(customer2['age'][customer2['ind_dela_fin_ult1'] == 1],
                        customer2['age'][customer2['ind_dela_fin_ult1'] == 0])

print("Anova 2 pValue  {0}".format(p2))
print("Anova 2 statistic  {0}".format(f2))

"""
pValue equal 0 which means that we can reject null hypothesis : 
So, customer age could be related to long term deposit
We can observe this fact also in the Boxplot
"""

# Anova test between  feature 'antiguedad' and 'ind_dela_fin_ult1'
# We need to test if it exists relationship between long term deposit and customer seniority
# The null hypothesis is the mean of customer seniority  between category of long term deposit is equal
f3, p3 = stats.f_oneway(customer2['antiguedad'][customer2['ind_dela_fin_ult1'] == 1],
                        customer2['antiguedad'][customer2['ind_dela_fin_ult1'] == 0])

print("Anova 3 pValue  {0}".format(p3))
print("Anova 3 statistic {0}".format(f3))

"""
pValue equal 0 which means that we can reject null hypothesis : 
So, customer seniority could be related to long term deposit
We can observe this fact also in the Boxplot
"""

# Boxplot Box Plot of gross income  by long term deposit
fig = plt.figure(figsize= (10, 10))
ax = fig.add_subplot(111)

ax.set_title("Box Plot of renta by long term deposit", fontsize= 20)
ax.set
data = [customer2['renta'][customer2['ind_dela_fin_ult1'] == 1],
        customer2['renta'][customer2['ind_dela_fin_ult1'] == 0]]

ax.boxplot(data,labels= ['1', '0', ],showmeans= True)

plt.xlabel("Long term deposit")
plt.ylabel("Gross income")
plt.show()

#----------------------------------------------------------
# Box Plot of Customer seniority by long term deposit
fig = plt.figure(figsize= (10, 10))
ax = fig.add_subplot(111)

ax.set_title("Box Plot of seniority by long term deposit", fontsize= 20)
ax.set
data = [customer2['antiguedad'][customer2['ind_dela_fin_ult1'] == 1],
        customer2['antiguedad'][customer2['ind_dela_fin_ult1'] == 0]]

ax.boxplot(data,labels= ['1', '0', ],showmeans= True)

plt.xlabel("Long term deposit")
plt.ylabel("Seniority")
plt.show()

#----------------------------------------------------------

# Box Plot of age by long term deposit
fig = plt.figure(figsize= (10, 10))
ax = fig.add_subplot(111)

ax.set_title("Box Plot of age by long term deposit", fontsize= 20)
ax.set
data = [customer2['age'][customer2['ind_dela_fin_ult1'] == 1],
        customer2['age'][customer2['ind_dela_fin_ult1'] == 0]]

ax.boxplot(data,labels= ['1', '0', ],showmeans= True)

plt.xlabel("Long term deposit")
plt.ylabel("Age")
plt.show()

# Anova test between  feature 'age' and 'ind_ecue_fin_ult1'
# We need to test if it exists relationship between eaccount and age
# The null hypothesis is the mean of age between category of eaccount is equal

f4, p4 = stats.f_oneway(customer2['age'][customer2['ind_ecue_fin_ult1'] == 1],
                        customer2['age'][customer2['ind_ecue_fin_ult1'] == 0])

print("Anova 4 pValue  {0}".format(p4))
print("Anova 4 statistic {0}".format(f4))

# Boxplot Box Plot of age by ind_ecue_fin_ult1
fig = plt.figure(figsize= (10, 10))
ax = fig.add_subplot(111)

ax.set_title("Box Plot of age by eaccount", fontsize= 20)
ax.set
data = [customer2['age'][customer2['ind_ecue_fin_ult1'] == 1],
        customer2['age'][customer2['ind_ecue_fin_ult1'] == 0]]

ax.boxplot(data,labels= ['1', '0', ],showmeans= True)

plt.xlabel("Eaccount")
plt.ylabel("Age")
plt.show()

"""
pValue equal 0 which means that we can reject null hypothesis : 
So having eaccount could be related to age
We can observe this fact also in the Boxplot
"""

# Save result dataset
#customer2.to_csv("cust_seg_Updated_NoScaler.csv")
customer2.to_csv("cust_seg_Updated.csv")

# Automated exploratory data analysis
report = sweetviz.analyze(customer2,pairwise_analysis='on')
#report.show_html(open_browser=True, filepath="D:/FOAD/DSTI/Data Glacer/Projects/final/EDA_Cust_Fin_NoScale.html", layout='widescreen')
report.show_html(open_browser=True, filepath="D:/FOAD/DSTI/Data Glacer/Projects/final/EDA_Cust_Fin.html", layout='widescreen')

"""
from sklearn.cluster import KMeans

kmean_data = pd.read_csv("cust_seg_3.csv", low_memory=False)

kmean_data = pd.get_dummies(kmean_data, columns=['fecha_dato','sexo', 'tiprel_1mes', 'canal_entrada','nomprov'])
kmean_data2=kmean_data[['age']]
kmean_data2 = kmean_data2.reset_index(drop=True)
print(kmean_data.info())
print(pd.isnull(kmean_data2).sum() > 0)
kmeans_model = KMeans(n_clusters=5, random_state=1).fit(kmean_data)
#kmeans_model.to
print(kmeans_model.labels_)
from numpy import savetxt
# define data
#data = asarray([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
# save to csv file
savetxt('data.csv', kmeans_model.labels_, delimiter=',')
"""