import numpy as np
import pandas as pd
from auxiliary_functions import check_categorical_columns, df_info, remove_blanks
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

df = pd.read_csv('moviereviews.tsv', sep='\t')

df_info(df)
check_categorical_columns(df)
remove_blanks(df)

print(f'The cleaned df has shape {df.shape}.')

X = df['review']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
text_clf = Pipeline([('tfIdf', TfidfVectorizer()), ('clf', LinearSVC())])
text_clf.fit(X_train, y_train)

predictions = text_clf.predict(X_test)

df_cm = pd.DataFrame(confusion_matrix(predictions, y_test), index=['pos', 'neg'], columns=['pos', 'neg'])
print(df_cm)
print('\n-----------------------------------------\n')
print(classification_report(predictions, y_test))
print('\n-----------------------------------------\n')
print(accuracy_score(predictions, y_test))
