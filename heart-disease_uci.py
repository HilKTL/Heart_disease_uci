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
import researchpy as rp
croostab = rp.crosstab(my_data['Thal'],my_data['Target'],test = 'chi-square')
print(croostab)
#There is a strong association between Thal and Target attribute.
# Thus, missing values are not completely at random,so missing values will be imputated with mode value of '3' of Thal attribute.
my_data['Thal'] = my_data['Thal'].replace('6.0', 6)
my_data['Thal'] = my_data['Thal'].replace('3.0', 3)
my_data['Thal'] = my_data['Thal'].replace('7.0', 7)
my_data['Thal'] = my_data['Thal'].replace('?', 3)

my_data['Thal']=my_data['Thal'].astype('category')
#my_data.info()

croostab = rp.crosstab(my_data['Ca'],my_data['Target'],test = 'chi-square')
print(croostab)
rp.crosstab(my_data['Ca'],my_data['Sex'],test = 'chi-square')
rp.crosstab(my_data['Ca'],my_data['Chestpain'],test = 'chi-square')
rp.crosstab(my_data['Ca'],my_data['FastingBloodSugar'],test = 'chi-square')
rp.crosstab(my_data['Ca'],my_data['RestECG'],test = 'chi-square')
rp.crosstab(my_data['Ca'],my_data['Exang'],test = 'chi-square')
rp.crosstab(my_data['Ca'],my_data['Slope'],test = 'chi-square')
rp.crosstab(my_data['Ca'],my_data['Thal'],test = 'chi-square')

"""Cramer's V coefficient association between Ca and categorical attributes                        
    
Ca-Sex : 0.13 , Ca-Chestpain : 0.185, Ca-FastingBloodSugar : 0.15 , Ca- RestECg : 0.12 , Ca-Exang :0.21 , Ca-Slope :0.14 , Ca-Thal : 0.2

There is no strong association between categorical values and ‘Ca’ attribute, so the association between numeric attributes and ‘Ca’ attribute will be evaluated with Kendall’s rank coefficient measurement.
"""

#Kendall’s rank coefficient (nonlinear) measurement
from scipy import stats

x1 = my_data['Ca']

x2 = my_data['MaxHeartRate']

tau,pvalue = stats.kendalltau(x1, x2)

print(tau,pvalue)

x1 = my_data['Ca']

x2 = my_data['Oldpeak']

tau, p_value = stats.kendalltau(x1, x2)

print(tau,p_value)

x1 = my_data['Ca']

x2 = my_data['Age']

tau, p_value = stats.kendalltau(x1, x2)

print(tau,p_value)

x1 = my_data['Ca']

x2 = my_data['RestingBloodPressure']

tau, p_value = stats.kendalltau(x1, x2)
print(tau,p_value)

mask = my_data[my_data['Ca'] == '?']

print(mask)
my_data['Ca'] = my_data['Ca'].replace('0.0', 0)
my_data['Ca'] = my_data['Ca'].replace('3.0', 3)
my_data['Ca'] = my_data['Ca'].replace('2.0', 2)
my_data['Ca'] = my_data['Ca'].replace('?', 0)
my_data['Ca'] = my_data['Ca'].replace('1.0', 1)

my_data.drop_duplicates(inplace = True)
len(my_data)
my_data.describe()

"""EXPLORATORY DATA ANALYSIS and VISUALIZATION

Questions to be explored

    Are there outlier values in this dataset?
    Is there any feature to be ignored because of weak association with labeled attribute?
    What is the most appropriate classification model that can be used for this data for future work?
    Which features have the most significant impact on causing heart disease?
    Is there any difference in the importance of attributes in terms of gender?

For numerical values ; Groupby() function is used to splitting data and describe() function is used to calculate the descriptive statistics of the data. The Pivot_table() method and count aggregation function is used to determine the number of patients having each unique values of categorical columns. For visualization; matplotlib and seaborn libraries are used.
"""

#statistical data of sick and non-sick patients;
targets = my_data.groupby('Target')
targets.describe().stack()
#The stack() function is used to reshape the dataframe from columns to index.

##statistical data of sick and non-sick patients in two genders;
targets1 = my_data.groupby(['Target','Sex'])
targets1.describe().stack()

targets = my_data.groupby('Target')
targets.mean()
targets = my_data.groupby(['Target','Chestpain'])
targets.size()
targets = my_data.groupby(['Target','Sex'])
targets.size()
#The Pivot_table() Method Count aggregation function will be used to determine the number of patients having each unique values of every column.
my_data.pivot_table(values ='New_ID' ,index = ['Sex','Chestpain','Target'],aggfunc = 'count')
my_data.pivot_table(values ='New_ID' ,index = ['Target','Thal','Sex'],aggfunc = 'count')

"""VISUALIZATION"""
import matplotlib.pyplot as plt
#the code below is necessary to see the visualization in the actual notebook itself,inline with the text.
import seaborn as sns

'''SEX'''
#• There are 303 patients of 206 male and 97 females; although the number of male patients is almost twice as female,
# the number of male patients is nearly four times the number of females in target 1.
targets = my_data.groupby(['Target','Sex'])
targets.size()

'''AGE'''
#• Age ranges from 29 to 77 with an average of 54.
# •Patients having heart disease (hd) are by majority between 57 to 60 years with an average of around 57 •
# In target 0 ,the average age is nearly 53. • Male patients are most likely to develop hd in younger age than females.
# The male patients with hd has an average age of 56 and female patients has an average age of 59

#https://seaborn.pydata.org/generated/seaborn.boxplot.html#seaborn.boxplot

sns.boxplot(x='Target',y='Age',hue ='Sex',data= my_data)

#• As seen in boxplot diagram ,there are outlier values in patients age .
# To get accurate mean values of age in target groups and genders,IQR method(Interquartile range method) will be applied to data..)
# The outlier value is the value that is over 1.5 times the interquartile, which is the difference between the third quartile and first quartile.
# • https://medium.com/analytics-vidhya/outlier-treatment-9bbe87384d02

#determination of outlier values
def outliers(x):
    q1 = my_data[x].quantile(0.25)
    q3 = my_data[x].quantile(0.75)
    Interquar=q3-q1
    print(q1)
    print(q3)
    print(Interquar)
    Lower_outlier_value = q1-(1.5*Interquar)
    Upper_outlier_value = q3+(1.5*Interquar)
    print(Lower_outlier_value, Upper_outlier_value)
outliers('Age')
my_data_age = my_data[my_data['Age'] > 80.5]
print(my_data_age.shape)
#there is no upper outlier value
my_data_age = my_data[my_data['Age'] < 28.5]
print(my_data_age.shape)
##there is no lower outlier value
targets['Age'].median()
sns.swarmplot(x='Age', y='Sex', data=my_data, hue='Target')

"""Chestpain"""
#Angina is chest pain that happens when blood arteries supplying blood to the heart are blocked.
#-Around 75% (105/139) of people have asymptomatic chest pain in target1, whereas only almost 25% (39/164) of people have asymptomatic chest pain
# in target 0.
#.While the number of male patients in target 0 was 21 out of 92 people, this figure reached 83 out of 114 people at target 1.
# In comparison it is seen that this number increased from 18  out of 72 people to 22 out of 25 people in females.
# Thus, asymptomatic chest pain has slightly higher correlation in male patients

targets = my_data.groupby('Target')
targets['Chestpain'].value_counts().plot(kind = 'bar',legend=True)
targets1 = my_data.groupby(['Target','Sex'])
targets1['Chestpain'].value_counts().plot(kind = 'bar',legend=True)
#to determine the number of patients of both gender and target groups
def target_genders(a,b) :
    x = my_data['Target']== a
    y = my_data['Sex']== b
    return my_data[x&y].shape[0]
number_female_targetone= target_genders(1,0)
number_female_targetzero= target_genders(0,0)
number_male_targetone= target_genders(1,1)
number_male_targetzero= target_genders(0,1)
#Determining the change in the incidence rate of chestpain 4 among the target groups
def chestpain_targets_genders(a,b,c) :
    x = my_data['Target']== a
    y = my_data['Sex']== b
    z = my_data['Chestpain']== c
    return my_data[x&y&z].shape[0]
chestpain_female_target0_percentage = chestpain_targets_genders(0,0,4)/number_female_targetzero*100
chestpain_female_target0_percentage
chestpain_female_target1_percentage = chestpain_targets_genders(1,0,4)/number_female_targetone*100
chestpain_female_target1_percentage
chestpain_male_target0_percentage = chestpain_targets_genders(0,1,4)/number_male_targetzero*100
chestpain_male_target0_percentage
chestpain_male_target1_percentage = chestpain_targets_genders(1,1,4)/number_male_targetone*100
chestpain_male_target1_percentage
#percentage ratio of two target groups with same gender
x = chestpain_female_target1_percentage/chestpain_female_target0_percentage
y = chestpain_male_target1_percentage/ chestpain_male_target0_percentage
result = [x,y]
result

chestpain_female_target0_percentage = chestpain_targets_genders(0,0,3)/number_female_targetzero*100
chestpain_female_target0_percentage
chestpain_female_target1_percentage = chestpain_targets_genders(1,0,3)/number_female_targetone*100
chestpain_female_target1_percentage
chestpain_male_target0_percentage = chestpain_targets_genders(0,1,3)/number_male_targetzero*100
chestpain_male_target0_percentage
chestpain_male_target1_percentage = chestpain_targets_genders(1,1,3)/number_male_targetone*100
chestpain_male_target1_percentage
#percentage ratio of two target groups with same gender
x = chestpain_female_target1_percentage/chestpain_female_target0_percentage
y = chestpain_male_target1_percentage/ chestpain_male_target0_percentage
result = [x,y]
result

chestpain_female_target0_percentage = chestpain_targets_genders(0,0,2)/number_female_targetzero*100
chestpain_female_target0_percentage
chestpain_female_target1_percentage = chestpain_targets_genders(1,0,2)/number_female_targetone*100
chestpain_female_target1_percentage
chestpain_male_target0_percentage = chestpain_targets_genders(0,1,2)/number_male_targetzero*100
chestpain_male_target0_percentage
chestpain_male_target1_percentage = chestpain_targets_genders(1,1,2)/number_male_targetone*100
chestpain_male_target1_percentage
#percentage ratio of two target groups with same gender
x = chestpain_female_target1_percentage/chestpain_female_target0_percentage
y = chestpain_male_target1_percentage/ chestpain_male_target0_percentage
result = [x,y]
result

"""SLOPE"""
#ST segment depression (horizontal or downsloping) is the most reliable indicator of exercise-induced ischaemia(a restriction in blood supply to tissues)

#-For patients in Target 1, the exercise-induced ST segment is mostly flat with a percentage of 65% (91 out of 139 patients), whereas, in Target 0, it is upsloping. (106 out of 164) -The percentage of slope flat values seen in women with illness is 2.1 times higher than women in target 0, while this value is 2.52 in men. Therefore, having flat slope has slightly more impact on males than females. -The percentage of down sloping values seen in women with illness is 4.32 times higher than women in target 0, while this value is 1.04 in men. Therefore, having flat slope has slightly more impact on males than females. -Having down sloping slope has more impact and females than males. .

def slope_targets_genders(d,e,f) :
    k = my_data['Target']== d
    l = my_data['Sex']== e
    m = my_data['Slope']== f
    return my_data[k&l&m].shape[0]

slope_female_target0_percentage = slope_targets_genders(0,0,2)/number_female_targetzero*100
print(slope_female_target0_percentage)
slope_female_target1_percentage = slope_targets_genders(1,0,2)/number_female_targetone*100
print(slope_female_target1_percentage)
slope_male_target0_percentage = slope_targets_genders(0,1,2)/number_male_targetzero*100
print(slope_male_target0_percentage)
slope_male_target1_percentage = slope_targets_genders(1,1,2)/number_male_targetone*100
print(slope_male_target1_percentage)
#percentage difference with same gender between two target groups
x = slope_female_target1_percentage/slope_female_target0_percentage
y = slope_male_target1_percentage/ slope_male_target0_percentage
result = [x,y]
print(result)

slope_female_target0_percentage = slope_targets_genders(0,0,3)/number_female_targetzero*100
print(slope_female_target0_percentage)
slope_female_target1_percentage = slope_targets_genders(1,0,3)/number_female_targetone*100
print(slope_female_target1_percentage)
slope_male_target0_percentage = slope_targets_genders(0,1,3)/number_male_targetzero*100
print(slope_male_target0_percentage)
slope_male_target1_percentage = slope_targets_genders(1,1,3)/number_male_targetone*100
print(slope_male_target1_percentage)
#percentage difference with same gender between two target groups
x = slope_female_target1_percentage/slope_female_target0_percentage
y = slope_male_target1_percentage/ slope_male_target0_percentage
result = [x,y]
print(result)

targets1 = my_data.groupby(['Target','Sex'])
targets1['Slope'].value_counts().plot(kind = 'bar',legend=True)
targets1['Slope'].value_counts()

