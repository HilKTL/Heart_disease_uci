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

for x in ['Target','Sex','Chestpain','FastingBloodSugar','RestECG','Exang','Slope','Thal']:
    croostab = rp.crosstab(my_data['Ca'],my_data['Target'],test = 'chi-square')
    print(croostab)
    print('\n--------------------------------')

"""Cramer's V coefficient association between Ca and categorical attributes                        
    
Ca-Sex : 0.13 , Ca-Chestpain : 0.185, Ca-FastingBloodSugar : 0.15 , Ca- RestECg : 0.12 , Ca-Exang :0.21 , Ca-Slope :0.14 , Ca-Thal : 0.2

There is no strong association between categorical values and ‘Ca’ attribute, so the association between numeric attributes and ‘Ca’ attribute will be evaluated with Kendall’s rank coefficient measurement.
"""

#Kendall’s rank coefficient (nonlinear) measurement
from scipy import stats

x1 = my_data['Ca']

for x in ['MaxHeartRate','Oldpeak','Age','Chol','RestingBloodPressure'] :
    x2 = my_data[x]

    tau,pvalue = stats.kendalltau(x1, x2)

    print(tau)
    print('\n-----------------------',x)

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
print(chestpain_female_target0_percentage)
chestpain_female_target1_percentage = chestpain_targets_genders(1,0,4)/number_female_targetone*100
print(chestpain_female_target1_percentage)
chestpain_male_target0_percentage = chestpain_targets_genders(0,1,4)/number_male_targetzero*100
print(chestpain_male_target0_percentage)
chestpain_male_target1_percentage = chestpain_targets_genders(1,1,4)/number_male_targetone*100
print(chestpain_male_target1_percentage)
#percentage ratio of two target groups with same gender
x = chestpain_female_target1_percentage/chestpain_female_target0_percentage
y = chestpain_male_target1_percentage/ chestpain_male_target0_percentage
result = [x,y]
result

chestpain_female_target0_percentage = chestpain_targets_genders(0,0,3)/number_female_targetzero*100
print(chestpain_female_target0_percentage)
chestpain_female_target1_percentage = chestpain_targets_genders(1,0,3)/number_female_targetone*100
print(chestpain_female_target1_percentage)
chestpain_male_target0_percentage = chestpain_targets_genders(0,1,3)/number_male_targetzero*100
print(chestpain_male_target0_percentage)
chestpain_male_target1_percentage = chestpain_targets_genders(1,1,3)/number_male_targetone*100
print(chestpain_male_target1_percentage)
#percentage ratio of two target groups with same gender
x = chestpain_female_target1_percentage/chestpain_female_target0_percentage
y = chestpain_male_target1_percentage/ chestpain_male_target0_percentage
result = [x,y]
print(result)

chestpain_female_target0_percentage = chestpain_targets_genders(0,0,2)/number_female_targetzero*100
print(chestpain_female_target0_percentage)
chestpain_female_target1_percentage = chestpain_targets_genders(1,0,2)/number_female_targetone*100
print(chestpain_female_target1_percentage)
chestpain_male_target0_percentage = chestpain_targets_genders(0,1,2)/number_male_targetzero*100
print(chestpain_male_target0_percentage)
chestpain_male_target1_percentage = chestpain_targets_genders(1,1,2)/number_male_targetone*100
print(chestpain_male_target1_percentage)
#percentage ratio of two target groups with same gender
x = chestpain_female_target1_percentage/chestpain_female_target0_percentage
y = chestpain_male_target1_percentage/ chestpain_male_target0_percentage
result = [x,y]
print(result)

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

"""Thal"""

"""Thalassemia is a blood disorder

• Around 64% (89 /139) people with heart disease have reversible defect type of thalassemia, represented by number 7.
 In comparison, around 17%( 28 out of 164) people in target 0 have reversible defect type of thalassemia. • 
 While only 2 out of 72 women who were not sick had a thal 7 value, this value increased from 13 out of 25 in sick women, 
 in comparison 26 out of 92 men who were not sick had a thal value of 7 and value was observed in 76 out of 114 patients in sick men. 
 Thus it is clear that reversible defect type of thalassemia has more impact on females than men having heart disease.
"""

my_data.pivot_table(values ='New_ID' ,index = ['Target','Thal','Sex'],aggfunc = 'count')
targets['Thal'].value_counts().plot(kind = 'bar' ,legend ='True')
targets1['Thal'].value_counts().plot(kind = 'bar',legend=True)

"""Ca

Ca (C-arm fluoroscopy results) C-arm is a medical imaging device Number of blocked major vessels supplying blood (0-3) colored by flouroscopy

• When target 0 and target 1 patient numbers are compared, the number of patients with 1 blocked vessel in target 1 increased more than 2 times, the number with 2 blocked vessels increased more than 4 times, and the number with 3 blocked vessels increased almost 6 times. • Female patients with three blocked vessels are all have heart disease. • Having two-blocked vessels has the greatest effect on both males and females in developing hd.But the percentage of having 2-blocked vessels in men with illness is 9.28 times higher than men in target 0, while this value is 4.6 in women.
"""
my_data.pivot_table(values ='New_ID' ,index = ['Target','Ca'],aggfunc = 'count')
my_data.pivot_table(values ='New_ID' ,index = ['Target','Ca','Sex'],aggfunc = 'count')
targets['Ca'].value_counts().plot(kind = 'bar',legend=True)
targets1['Ca'].value_counts().plot(kind = 'bar',legend=True)
def target_genders(a,b) :
    x = my_data['Target']== a
    y = my_data['Sex']== b
    return my_data[x&y].shape[0]
number_female_targetone= target_genders(1,0)
number_female_targetzero= target_genders(0,0)
number_male_targetone= target_genders(1,1)
number_male_targetzero= target_genders(0,1)
#Determining the change in the incidence rate of ca 1 among the target groups
def ca_targets_genders(a,b,c) :
    x = my_data['Target']== a
    y = my_data['Sex']== b
    z = my_data['Ca']== c
    return my_data[x&y&z].shape[0]
ca_female_target0_percentage = ca_targets_genders(0,0,1)/number_female_targetzero*100
print(ca_female_target0_percentage)
ca_female_target1_percentage = ca_targets_genders(1,0,1)/number_female_targetone*100
print(ca_female_target1_percentage)
ca_male_target0_percentage = ca_targets_genders(0,1,1)/number_male_targetzero*100
print(ca_male_target0_percentage)
ca_male_target1_percentage = ca_targets_genders(1,1,1)/number_male_targetone*100
print(ca_male_target1_percentage)
#percentage ratio of two target groups with same gender
x = ca_female_target1_percentage/ca_female_target0_percentage
y = ca_male_target1_percentage/ ca_male_target0_percentage
result = [x,y]
print(result)

#Determining the change in the incidence rate of ca 2 among the target groups
ca_female_target0_percentage = ca_targets_genders(0,0,2)/number_female_targetzero*100
print(ca_female_target0_percentage)
ca_female_target1_percentage = ca_targets_genders(1,0,2)/number_female_targetone*100
print(ca_female_target1_percentage)
ca_male_target0_percentage = ca_targets_genders(0,1,2)/number_male_targetzero*100
print(ca_male_target0_percentage)
ca_male_target1_percentage = ca_targets_genders(1,1,2)/number_male_targetone*100
print(ca_male_target1_percentage)
#percentage ratio of two target groups with same gender
x = ca_female_target1_percentage/ca_female_target0_percentage
y = ca_male_target1_percentage/ ca_male_target0_percentage
result = [x,y]
print(result)
#https://seaborn.pydata.org/tutorial/categorical.html
sns.catplot(y="Ca", hue="Target", kind="count",data=my_data)
plt.show()

"""Oldpeak

ST depression induced by exercise relative to rest

• For patients in target 1, the avg. old peak value of 1,41 is 2,5 times the avg. old peak value in patients of target 0.(0,56) 
• Males do not show disease at higher old peak values in target 0 compared to female patients.In target 1 ,
males have higher mean value of 1.42 than females with a mean value of 1.36.
"""

sns.distplot(my_data[my_data['Target']==1]['Oldpeak'],kde=True,bins=35)
sns.distplot(my_data[my_data['Target']==0]['Oldpeak'],kde=True,bins=35,label =['Target 0'])
#https://seaborn.pydata.org/generated/seaborn.boxplot.html#seaborn.boxplot
sns.boxplot(x='Target',y='Oldpeak',hue ='Sex',data= my_data)
outliers('Oldpeak')
#determination of outlier values
q1=my_data['Oldpeak'].quantile(0.25)
q3=my_data['Oldpeak'].quantile(0.75)
Interquar=q3-q1
print(q1)
print(q3)
print(Interquar)
Lower_outlier_value = q1-(1.5*Interquar)
Upper_outlier_value = q3+(1.5*Interquar)
print(Lower_outlier_value, Upper_outlier_value)
my_data_oldpeak = my_data[my_data['Oldpeak'] > Upper_outlier_value]
print(my_data_oldpeak)
my_data_oldpeak1=my_data[my_data['Oldpeak'] < Upper_outlier_value]
my_data_oldpeak1.describe()
mask = my_data_oldpeak1.groupby(['Target','Sex'])
mask.describe().stack()

"""Restecg

Resting ecg results

• 80 out of 139 ( around 57%) sick patients are showing probable or definite left ventricular hypertrophy by Estes' criteria(value2). 
In contrast, it is 68 out of 164.(around 41%) in target 0
• The percentage of sick women with probable or definite left ventricular hypertrophy is almost 1.30 times higher than 
the percentage of non-sick women having this value,in comparison the percentage of it is nearly 1.44 times higher in men 
than in those who are not sick.Thus, having probable or definite left ventricular hypertrophy has more impact on males than females.
"""
targets['RestECG'].value_counts().plot(kind = 'bar',legend=True)
targets1['RestECG'].value_counts().plot(kind = 'bar',legend=True)
def restecg_targets_genders(a,b,c) :
    x = my_data['Target']== a
    y = my_data['Sex']== b
    z = my_data['RestECG']== c
    return my_data[x&y&z].shape[0]
restecg2_female_target0_percentage = restecg_targets_genders(0,0,2)/number_female_targetzero*100
print(restecg2_female_target0_percentage)
restecg2_female_target1_percentage = restecg_targets_genders(1,0,2)/number_female_targetone*100
print(restecg2_female_target1_percentage)
restecg2_male_target0_percentage = restecg_targets_genders(0,1,2)/number_male_targetzero*100
print(restecg2_male_target0_percentage)
restecg2_male_target1_percentage = restecg_targets_genders(1,1,2)/number_male_targetone*100
print(restecg2_male_target1_percentage)
#percentage difference with same gender between two target groups
x = restecg2_female_target1_percentage/restecg2_female_target0_percentage
y = restecg2_male_target1_percentage/ restecg2_male_target0_percentage
result = [x,y]
print(result)
targets1['RestECG'].value_counts()

"""Exang

(exercise induced angina)

• 76 out of 139 people in target 1 have exercise-induced angina. In comparison, it is 23 out of 161 in target 0.
 • The percentage of women with exercise-induced angina at target 1 is 5 times the percentage at target 0 and for men it is 3.3 times 
 the percentage at target 0 . • Having exercise-induced angina has greater impact on females in developing heart disease than men.
"""
targets1['Exang'].value_counts().plot(kind = 'barh',legend=True)
targets['Exang'].value_counts().plot(kind = 'barh',legend=True)
#to determine the number of patients of different gender in target groups
def exang_targets_genders(a,b,c) :
    x = my_data['Target']== a
    y = my_data['Sex']== b
    z = my_data['Exang']== c
    return my_data[x&y&z].shape[0]
exang_female_target0_percentage = exang_targets_genders(0,0,1)/number_female_targetzero*100
print(exang_female_target0_percentage)
exang_female_target1_percentage = exang_targets_genders(1,0,1)/number_female_targetone*100
print(exang_female_target1_percentage)
exang_male_target0_percentage = exang_targets_genders(0,1,1)/number_male_targetzero*100
print(exang_male_target0_percentage)
exang_male_target1_percentage = exang_targets_genders(1,1,1)/number_male_targetone*100
print(exang_male_target1_percentage)
#percentage difference with same gender between two target groups
x = exang_female_target1_percentage/exang_female_target0_percentage
y = exang_male_target1_percentage/ exang_male_target0_percentage
result = [x,y]
print(result)

"""Chol

Serum cholesterol in mg/dl

• The difference in cholesterol levels between those who are sick and those who are not is about 10 mmHg, and this value 
is about 17 mmHg in women and 15 mmHg in men. • Men develop disease with lower cholesterol levels than women.
"""
sns.distplot(my_data[my_data['Target']==1]['Chol'],kde=True,bins=35)
plt.show()
sns.distplot(my_data[my_data['Target']==0]['Chol'],kde=True,bins=35)
plt.show()
sns.boxplot(x='Target',y='Chol',hue ='Sex',data= my_data)
#https://medium.com/analytics-vidhya/outlier-treatment-9bbe87384d02
outliers('Chol')
mask = my_data['Chol'] < 371
mask1 = my_data['Chol'] > 115
my_data_chol= my_data[mask&mask1]
my_data_chol.describe()
mask = my_data_chol.groupby(['Target','Sex'])
mask.describe().stack()
sns.swarmplot(x='Chol', y='Thal', data=my_data_chol, hue='Target')
plt.show()
sns.boxplot(x='Target',y='Chol',hue ='Chestpain',data= my_data_chol)
plt.show()

"""Maximum Heart Rate

By convention, the maximum predicted heart rate is calculated as 220 (210 for women) minus the patient’s age.

• Sick patients achieved less heart rate with an avg. of 139 than non sick patients with average value of 158 . 
• In target 0,the avg.heart rate of men is almost 162 and it is 154 for women. 
• In target 1,the avg.heart rate of men is nearly 138 and it is nearly 143 for women. 
• Women are more susceptible to heart rate decline in developing the disease.
"""

outliers('MaxHeartRate')
my_data_maxheartrate = my_data[my_data['MaxHeartRate'] > Upper_outlier_value]
print(my_data_maxheartrate.shape)
#there is no upper outlier value in maximum heart rate column
my_data_maxheartrate1 = my_data[my_data['MaxHeartRate'] < 84.75]
print(my_data_maxheartrate1)
##There is only one lower outlier value so it was ignored as it had no meaningful effect on the mean value.

targets = my_data.groupby(['Target'])
targets.mean()
targets.describe().stack()
mask = my_data.groupby(['Target','Sex'])
mask['MaxHeartRate'].mean()
sns.distplot(my_data[my_data['Target']==1]['MaxHeartRate'],kde=True,bins=35,rug=True)
sns.distplot(my_data[my_data['Target']==0]['MaxHeartRate'],kde=True,bins=35,rug=True)
"""Resting Blood Pressure

in mm Hg on admission to hospital

• The average blood pressure is 128 mmHg in non-sick patients, and it is 131 mmHg in patients with heart disease. • In target 1, the avg. rbp level for women is 138.15 mmHg, it is lower in men with a value of 130.2 mmHg. • In target 1, male patients' third quartile rbp value is 140 mmHg, whereas it is 150 mmHg in females. • Patients having resting blood pressure above 180 mmHg are all have heart disease. • It is observed that men develop heart disease in lower blood pressure than women.
"""
outliers('RestingBloodPressure')
my_data_restingbloodpressure = my_data[my_data['RestingBloodPressure'] > 170]
print(my_data_restingbloodpressure.shape)
my_data_restingbloodpressure1 = my_data[my_data['RestingBloodPressure'] < 90]
print(my_data_restingbloodpressure1.shape)
my_data_restingbloodpressure1 = my_data[my_data['RestingBloodPressure'] < 90]
print(my_data_restingbloodpressure1.shape)
my_data_rbp = my_data[my_data['RestingBloodPressure'] < 170]
my_data_rbp.describe()
mask = my_data_rbp.groupby(['Target','Sex'])
mask.describe().stack()
sns.distplot(my_data[my_data['Target']==1]['RestingBloodPressure'],kde=True,bins=35,rug=True)
sns.distplot(my_data[my_data['Target']==0]['RestingBloodPressure'],kde=True,bins=35,rug=True)

"""Fasting Blood Sugar

• In target 1, 22 out of 139 patients have high fbs level with a percentage of 15.8% and in target 0,the percentage of having high fbs level is 14.0%. • The percentage of having high fbs levels in women with illness is 2.88 times higher than women in target 0, while this value is 0.76 times lower in men. • High fbs level has a greater effect in women than in men.
"""
targets['FastingBloodSugar'].value_counts().plot(kind = 'barh',legend=True)
targets['FastingBloodSugar'].value_counts()
def target_genders(a,b) :
    x = my_data['Target']== a
    y = my_data['Sex']== b
    return my_data[x&y].shape[0]
number_female_targetone= target_genders(1,0)
number_female_targetzero= target_genders(0,0)
number_male_targetone= target_genders(1,1)
number_male_targetzero= target_genders(0,1)
def fbs_targets_genders(d,e,f) :
    k = my_data['Target']== d
    l = my_data['Sex']== e
    m = my_data['FastingBloodSugar']== f
    return my_data[k&l&m].shape[0]
fbs_female_target0_percentage = fbs_targets_genders(0,0,1)/number_female_targetzero*100
print(fbs_female_target0_percentage)
fbs_female_target1_percentage = fbs_targets_genders(1,0,1)/number_female_targetone*100
print(fbs_female_target1_percentage)
fbs_male_target0_percentage = fbs_targets_genders(0,1,1)/number_male_targetzero*100
print(fbs_male_target0_percentage)
fbs_male_target1_percentage = fbs_targets_genders(1,1,1)/number_male_targetone*100
print(fbs_male_target1_percentage)
#percentage difference with same gender between two target groups
x = fbs_female_target1_percentage/fbs_female_target0_percentage
y = fbs_male_target1_percentage/ fbs_male_target0_percentage
result = [x,y]
print(result)
targets = my_data.groupby(['Target','Sex','FastingBloodSugar'])
targets.size()

"""FEATURE SELECTION
Determining the association between Target attribute and categorical attributes:
Cramer coefficient correlation between categorical attributes :
"""
#https://researchpy.readthedocs.io/en/latest/crosstab_documentation.html
import researchpy
for x in ['Sex','RestECG','Exang','Slope','Thal','FastingBloodSugar','Chestpain'] :
    croostab = researchpy.crosstab(my_data['Target'],my_data[x],test = 'chi-square')
    print(croostab)
    print('\n------------------------------------')

"""According to Cramer's V correlation ; Having thalassemia disorder and having angina have strong correlation with developing heart disease.
C-arm fluoroscopy results,having exercise-induced angina and Slope results have moderate association with developing heart disease.
Fasting Blood sugar level has the lowest association with target 1."""

"""Determining the association between Target attribute and numerical attributes with Logistic Regression"""

#https://towardsdatascience.com/interpreting-coefficients-in-linear-and-logistic-regression-6ddf1295f6f1
from sklearn. linear_model import LogisticRegression
X = my_data[['Age','RestingBloodPressure','Chol','MaxHeartRate','Oldpeak']]
y = my_data['Target']
logreg = LogisticRegression(solver = 'liblinear')
logreg.fit(X,y)
y = y.astype('float')
log_odds = logreg.coef_[0]
pd.DataFrame(log_odds,X.columns,columns = ['coef'])\
.sort_values(by='coef',ascending=False)
"""According to results:

Maximum heart rate achieved and ST depression induced by exercise relative to rest have the strongest association with developing heart disease.Cholesterol levels and Age and Resting Blood Pressure have weaker association with developing heart disease.
"""

"""APPLY MODEL"""
#MODEL SELECTION
from sklearn import model_selection
from sklearn. linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
data = my_data
values = data.values
# Preparing Data For Training
Y = values[:,14]
#target column is labeled.
X = values[:,0:14]
Y= Y.astype('int32')

"""Cross validation without removing outlier values and weak associated features"""
#Cross validation (Model evaluation)
outcome = []
model_names = []
models = [('LogReg', LogisticRegression(solver = 'liblinear')),
          ('GaussianNB', GaussianNB()),
          ('RandomForest',RandomForestClassifier(n_estimators=100)),
          ('DecTree', DecisionTreeClassifier()),
          ('KNN', KNeighborsClassifier())]
#Cross validation (Model evaluation)
random_seed =12
for model_name, model in models:
    k_fold_validation = model_selection.KFold(n_splits=10, random_state=random_seed,shuffle = True)
    results = model_selection.cross_val_score(model, X, Y, cv=k_fold_validation, scoring='accuracy')
    outcome.append(results)
    model_names.append(model_name)
    output_message = "%s| Mean=%f STD=%f" % (model_name, results.mean(), results.std())
    print(output_message)
#Visualization of performances of models
fig = plt.figure()
fig.suptitle('ML Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(outcome)
ax.set_xticklabels(model_names)
plt.show()
"""Removing Outliers"""
outliers('Oldpeak')
my_data.drop(my_data[my_data['Oldpeak'] > 4].index, inplace = True)
print(my_data.shape)
my_data.drop(my_data[my_data['Chol'] > 371].index, inplace = True)
print(my_data.shape)
outliers('MaxHeartRate')
my_data.drop(my_data[my_data['MaxHeartRate'] < 84.75].index, inplace = True)
print(my_data.shape)
outliers('RestingBloodPressure')
my_data.drop(my_data[my_data['RestingBloodPressure'] > 170].index, inplace = True)
print(my_data.shape)
"""Cross validation after removing outlier values"""
#MODEL COMPARISON AFTER REMOVING OUTLIERS
random_seed =12
for model_name, model in models:
    k_fold_validation = model_selection.KFold(n_splits=10, random_state=random_seed,shuffle=True)
    results = model_selection.cross_val_score(model, X, Y, cv=k_fold_validation, scoring='accuracy')
    outcome.append(results)
    model_names.append(model_name)
    output_message = "%s| Mean=%f STD=%f" % (model_name, results.mean(), results.std())
    print(output_message)
fig = plt.figure()
fig.suptitle('Machine Learning Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(outcome)
ax.set_xticklabels(model_names,rotation= 45)

plt.show()
#After removing outliers; the performance of Decision Tree increased.

"""Cross validation after removing outlier values and weak associated features"""
my_datas = my_data.drop(columns = ['Chol','FastingBloodSugar','Age','RestingBloodPressure'])
print(my_datas.shape)
data = my_datas
values = data.values

Y = values[:,10]
X = values[:,0:10]
Y= Y.astype('int32')
random_seed =12
for model_name, model in models:
    k_fold_validation = model_selection.KFold(n_splits=10, random_state=random_seed,shuffle =True)
    results = model_selection.cross_val_score(model, X, Y, cv=k_fold_validation, scoring='accuracy')
    outcome.append(results)
    model_names.append(model_name)
    output_message = "%s| Mean=%f STD=%f" % (model_name, results.mean(), results.std())
    print(output_message)

#https://towardsdatascience.com/interpreting-coefficients-in-linear-and-logistic-regression-6ddf1295f6f1
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
X = my_data[['Age','Chestpain','Sex','RestingBloodPressure','RestECG','MaxHeartRate','Exang','Oldpeak',
         'Slope','Ca','Thal','Chol','FastingBloodSugar']]
y = my_data['Target']
logreg = LogisticRegression(solver = 'liblinear')
logreg.fit(X,y)
log_odds = logreg.coef_[0]

pd.DataFrame(log_odds,X.columns,columns = ['coef'])\
.sort_values(by='coef',ascending=False)
odds = np.exp(logreg.coef_[0])
pd.DataFrame(odds,
             X.columns,
             columns=['coef'])\
            .sort_values(by='coef', ascending=False)
#FEATURE IMPORTANCE IN FEMALES
my_datafem = my_data[my_data['Sex']== 0]
#my_datafem.head()
X = my_datafem[['Age','Chestpain','RestingBloodPressure','RestECG','MaxHeartRate','Exang','Oldpeak',
         'Slope','Ca','Thal','Chol','FastingBloodSugar']]
y = my_datafem['Target']
logreg = LogisticRegression(solver = 'liblinear')
logreg.fit(X,y)
log_odds = logreg.coef_[0]

pd.DataFrame(log_odds,X.columns,columns = ['coef'])\
.sort_values(by='coef',ascending=False)
odds = np.exp(logreg.coef_[0])
pd.DataFrame(odds,
             X.columns,
             columns=['coef'])\
            .sort_values(by='coef', ascending=False)
#FEATURE IMPORTANCE IN MALES
my_data_male = my_data[my_data['Sex']== 1]
#my_data_male.head()
X = my_data_male[['Age','Chestpain','RestingBloodPressure','RestECG','MaxHeartRate','Exang','Oldpeak',
         'Slope','Ca','Thal','Chol','FastingBloodSugar']]
y = my_data_male['Target']
logreg = LogisticRegression(solver = 'liblinear')
logreg.fit(X,y)
log_odds = logreg.coef_[0]

pd.DataFrame(log_odds,X.columns,columns = ['coef'])\
.sort_values(by='coef',ascending=False)
odds = np.exp(logreg.coef_[0])
pd.DataFrame(odds,
             X.columns,
             columns=['coef'])\
            .sort_values(by='coef', ascending=False)
