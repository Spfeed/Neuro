import pandas as pd
import numpy as np

titanic=pd.read_csv(filepath_or_buffer="C:/Users/subba/Desktop/datasets/titanic.csv",
            sep=',')
male=len(titanic[titanic['Sex']=='male'])
female=len(titanic[titanic['Sex']=='female'])

print('man: ', male,',' ' woman: ', female)

passengerscount=len(titanic['PassengerId'])

survived=len(titanic[titanic['Survived']==1])
survivors=round((survived/passengerscount)*100,2)
print('passengers survived in percentage: ', survivors)

firstclass=round((len(titanic[titanic['Pclass']==1])/passengerscount)*100,2)
print('passengers of 1 class: ', firstclass)

mediumage=titanic['Age'].sum()/passengerscount
median=titanic['Age'].median()
print('medium age: ',mediumage, ', median: ',median)

correl=titanic['SibSp'].corr(titanic['Parch'])
print('correlation: ',correl)

test=titanic[titanic['Sex']=='female']

def extract_female_name(name):
    if '(' in name and ')' in name:

        return name.split('(')[1].split(')')[0]
    elif 'Miss.' in name:

        return name.split('Miss. ')[1].split(' ')[0]
    else:

        return None

test['FemaleName']=test['Name'].apply(extract_female_name)
name_counts=test['FemaleName'].value_counts()
name_counts=name_counts.dropna()
most_popular_name=name_counts.idxmax()
print(most_popular_name)