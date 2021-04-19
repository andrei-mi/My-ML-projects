import numpy as np
import pandas as pd
from NLP.auxiliary_functions import df_info, check_categorical_columns, remove_blanks
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

df = pd.read_csv('../Datasets/moviereviews.tsv', sep='\t')

# Display some initial information about the dataframe
df_info(df)

# Check if the categorical columns have empty/space strings
check_categorical_columns(df)

# Remove the rows containing space strings in the review column
remove_blanks(df)

print(f'The cleaned df has shape {df.shape}.')

sid = SentimentIntensityAnalyzer()

df['compound'] = df['review'].apply(lambda review: sid.polarity_scores(review)['compound'])
df['compound_label'] = df['compound'].apply(lambda score: 'pos' if score >= 0 else 'neg')

# Display metrics used to evaluate the model
df_cm = pd.DataFrame(confusion_matrix(df['label'], df['compound_label']), index=['pos', 'neg'], columns=['pos', 'neg'])
print('\n-----------------------------------------\n')
print(f'The confusion matrix is:\n\n{df_cm}')
print('\n-----------------------------------------\n')
print(f"The classification report is:\n\n{classification_report(df['label'], df['compound_label'])}")
print('\n-----------------------------------------\n')
print(f"The accuracy of the model is {round(100*accuracy_score(df['label'], df['compound_label']),2)}%.")
