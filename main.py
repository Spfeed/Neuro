import pandas as pd
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier

#1 задание
titanic=pd.read_csv(filepath_or_buffer="datasets/titanic.csv",
            sep=',')
#5 задние - сразу для удобства исключим пустые значения
titanic=titanic.dropna()

#2 задание
tit_2=titanic[['Pclass','Fare','Age','Sex']].copy()

#3  задание
tit_2['Sex']=tit_2['Sex'].replace({'female': 0, 'male': 1})

#4 задание
aim_var=titanic['Survived']

#6 задание
clf=DecisionTreeClassifier(random_state=241)
clf.fit(tit_2,aim_var)

#7 задание
importances=clf.feature_importances_
feature_importance_dict = dict(zip(tit_2.columns, importances))
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
top_two_features = sorted_feature_importance[:2]
print("Два признака с наибольшей важностью:")
for feature, importance in top_two_features:
    print(f"{feature}: {importance}")

#8 задание
person_data={
    'Pclass': 1,
    'Fare': 27.7208,
    'Age': 39,#3 не выживет((((
    'Sex': 0
}
person_df=pd.DataFrame([person_data])
survival_prediction=clf.predict(person_df)
if survival_prediction[0]==0:
    print("Человек не выжил")
else:
    print("Человек выжил")
