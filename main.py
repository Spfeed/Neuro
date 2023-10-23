from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np

newsgroups = fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])

# Вычисление TF-IDF признаков для текстов
tfidf_vectorizer = TfidfVectorizer(max_df=0.5, stop_words='english')
X = tfidf_vectorizer.fit_transform(newsgroups.data)

# Подбор лучшего параметра C для SVM с линейным ядром с помощью кросс-валидации
param_grid = {'C': [10 ** i for i in range(-5, 6)]
}
svm = SVC(kernel='linear', random_state=241)
kfold = KFold(n_splits=5, shuffle=True ,random_state=241)
grid_search = GridSearchCV(svm, param_grid, cv=kfold, scoring='accuracy')
grid_search.fit(X, newsgroups.target)
best_C = grid_search.best_params_['C']

# Обучение SVM на всей выборке с лучшим параметром C
svm = SVC(kernel='linear', C=best_C, random_state=241)
svm.fit(X, newsgroups.target)

# Получение веса признаков
coef = svm.coef_.toarray()[0]

# Поиск 10 слов с наибольшими по модулю весами
top_indices = np.argsort(np.abs(coef))[::-1][:10]
feature_names = tfidf_vectorizer.get_feature_names_out()
top_words = [feature_names[i] for i in top_indices]

# Вывод найденнх слов
print(top_words)
