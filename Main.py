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


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

## Data loading and exploratory data analysis

os.chdir("D:/FOAD/DSTI/Data Glacer/Projects/final/")
customer = pd.read_csv("cust_seg.csv", low_memory=False)

#Print dataset structure
print(customer.info())

# Missing values checking

# Total missing value
print(customer.isnull().values.sum())

# Missing value as per column
print(customer.isnull().sum(axis=0))

# Correlation matrix

correlations = customer.corr()
correlations.to_csv("corr.csv")

sns.heatmap(correlations)
plt.show()

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
customer = customer.drop('fecha_alta',1)

#NA dropping
#customer2 = customer.dropna(subset=['sexo', 'fecha_alta'])
#customer2 = customer.dropna(subset=['sexo'])

#Data conversion
customer.age =  pd.to_numeric(customer.age, errors="coerce")
customer.antiguedad =  pd.to_numeric(customer.antiguedad, errors="coerce")

#customer2.fecha_alta =  pd.Categorical(customer2.fecha_alta)
#customer2.canal_entrada =  pd.Categorical(customer2.canal_entrada)

#Outlear dropping
customer3 = customer.drop(customer[customer.antiguedad < 0].index)

# Manage missing value in numerical variable
customer3['renta'] = customer3['renta'].fillna((customer3['renta'].mean()))

# Manage missing value in categorical variable
customer3['sexo'] = customer3['sexo'].fillna("U")
customer3['canal_entrada'] = customer3['canal_entrada'].fillna("Unknown")
customer3.canal_entrada =  pd.Categorical(customer3.canal_entrada)
customer3['nomprov'] = customer3['nomprov'].fillna("Unknown")
customer3['ind_nom_pens_ult1'] = customer3['ind_nom_pens_ult1'].fillna(3)

# Data scaling for numerical variable
# In the case of column renta, we first use log transformation
# due to the great disparity (right skewness)  of data in that column

sc = StandardScaler()

customer3.loc[:,['age']] = sc.fit_transform(customer3.loc[:,['age']])
customer3.loc[:,['antiguedad']] = sc.fit_transform(customer3.loc[:,['antiguedad']])

customer3.loc[:, ['renta']] = np.log(customer3.loc[:,['renta']])

customer3.loc[:, ['renta']] = sc.fit_transform(customer3.loc[:, ['renta']])

#Dropping duplicated rows after some preprocessing
customer4 = customer3.drop_duplicates()

# Fix index after remove duplicate
customer4 = customer4.reset_index(drop=True)

# Data structure after preprocessing
print(customer4.info())

# Save result dataset
customer4.to_csv("cust_seg_2.csv")

# Automated explotaory data analysis
report = sweetviz.analyze(customer4,pairwise_analysis='on')
report.show_html(open_browser=True, filepath="D:/FOAD/DSTI/Data Glacer/Projects/final/EDA_Cust_Final.html", layout='widescreen')






