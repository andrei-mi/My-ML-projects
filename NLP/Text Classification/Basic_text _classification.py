import pandas as pd
from auxiliary_functions import check_categorical_columns, df_info, remove_blanks, processed_auroc
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

df = pd.read_csv('moviereviews.tsv', sep='\t')

# Display some initial information about the dataframe
df_info(df)

# Check if the categorical columns have empty/space strings
check_categorical_columns(df)

# Remove the rows containing space strings in the review column
remove_blanks(df)

print(f'The cleaned df has shape {df.shape}.')

# Preparation for modelling
X = df['review']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Building a classification pipeline object which transforms the text data using TF-IDF into an understandable numerical input for a classifier
text_clf = Pipeline([('tfIdf', TfidfVectorizer()), ('clf', LinearSVC())])
text_clf.fit(X_train, y_train)

# Make predictions on the test data
predictions = text_clf.predict(X_test)

# Display metrics used to evaluate the model
df_cm = pd.DataFrame(confusion_matrix(predictions, y_test), index=['pos', 'neg'], columns=['pos', 'neg'])
print('\n-----------------------------------------\n')
print(f'The confusion matrix is:\n\n{df_cm}')
print('\n-----------------------------------------\n')
print(f'The classification report is:\n\n{classification_report(predictions, y_test)}')
print('\n-----------------------------------------\n')
print(f'The accuracy of the model on the test data is {round(100*accuracy_score(predictions, y_test),2)}%.')
print('\n-----------------------------------------\n')
print(f'The AUROC of the model on the test data is {round(processed_auroc(y_test, predictions), 3)}.')
