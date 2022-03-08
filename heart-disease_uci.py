#Pulling dataset from 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/ '
import requests
file = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data")
heart_dis = file.text
#read file and assign it to variable
#give write permission to the opened file
#write the contents of the heart_dis variable to the file, then close the file by appyling the changes
hd = open("heart_dis.csv", "w")
hd.write(heart_dis)
hd.close()
import pandas as pd
my_data = pd.read_csv('C:\\Users\\serta\\Desktop\\my_data\\heart_dis.csv',header=None)
print(my_data.head(2))

"""DATA WRANGLING"""
#Naming columns
my_data.columns =['Age','Sex','Chestpain','RestingBloodPressure','Chol','FastingBloodSugar','RestECG','MaxHeartRate',
                  'Exang','Oldpeak','Slope','Ca','Thal','Health Condition']
print(my_data.head(2))
#Adding new_column
my_data.insert(0, 'New_ID', range(1, 1 + len(my_data)))
print(my_data.head(2))

"""Creating Target Column from Health Condition Values """
#'0' MEANS ABSENCE OF HEART DISEASE, '1,2,3,4' VALUES REPRESENT PRESENCE OF THE DISEASE

my_data['Health Condition'].unique()
my_data['Target'] = [0 if x == 0 else 1 for x in my_data['Health Condition']]
print(my_data.head(2))
#Deleting column
my_data.drop(['Health Condition'],axis=1,inplace = True)
print(my_data.head(2))
#concise summary of a DataFrame
my_data.info()

"""Checking duplicated rows"""
duplicate_rows = my_data[my_data.duplicated()]
print(duplicate_rows.shape)

"""According to info method there is no null value but to be sure unique values will be checked"""

my_data['Chestpain'].unique()
my_data['Ca'].unique()
my_data['Thal'].unique()
my_data['Sex'].unique()
my_data['RestECG'].unique()
my_data['FastingBloodSugar'].unique()
my_data['Exang'].unique()
my_data['Slope'].unique()
my_data['Oldpeak'].unique()
my_data['Age'].unique()
my_data['RestingBloodPressure'].unique()
my_data['Chol'].unique()

"""Finding number of '?' values in Ca and Thal attributes:"""

my_data['Thal'].value_counts()
my_data['Ca'].value_counts()

"""HANDLING MISSING DATA"""
#DETERMINING THE PERCENTAGE OF MISSING ROWS IN DATAFRAME :
missing_data1 = len(my_data[my_data['Ca']=='?'])
print(missing_data1)
missing_data2 = len(my_data[my_data['Thal']=='?'])
print(missing_data2)
sum_missing = missing_data1 + missing_data2
total_columns =303
percent_missing_data = (sum_missing/total_columns) * 100
print(percent_missing_data)

#Because the percentage of missing data is lower than 2%, missing data will be evaluated to discover if it is missing completely at random or not.
# If missing values are MCAR, values will be deleted to avoid bias. Otherwise, data imputation will be made.

"""Determination of the datatypes of columns:"""
my_data.info()
# IN ORDER TO HAVE VALID RESULTS: Datatypes of categorical and numerical attributes will be corrected.
category_features =['Chestpain','FastingBloodSugar','RestECG','Exang','Thal','Sex','Target','Ca','New_ID','Slope']
my_data[category_features] = my_data[category_features].astype('category')
num_features = ['Age','RestingBloodPressure','Chol','MaxHeartRate']
my_data[num_features]= my_data[num_features].astype('int')

my_data.info()

"""FEATURE SELECTION

Because the percentage of missing data is lower than 2%, missing data will be evaluated to discover if missing data is completely at random or not. 
If missing values are MCAR, values will be deleted to avoid bias. Otherwise, data imputation will be made.[5] 
https://towardsdatascience.com/statistical-test-for-mcar-in-python-9fb617a76eac 
For determining the correlation between categorical variables cramer’s v measurement will be used which is based on a nominal variation of pearson’s Chi-square test.
https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9 
For determination of association between Ca and continuous variables Kendall’s rank coefficient (nonlinear) measurement will be used 
because as an input Ca attribute’s datatype is ordinal and output variables are numerical. 
https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/ 
https://www.statstutor.ac.uk/resources/uploaded/tutorsquickguidetostatistics.pdf
"""

"""Cramer Coefficient Correlation Between Categorical Attributes :"""
