import numpy as np
import pandas as pd
from auxiliary_functions import check_categorical_columns, df_info, remove_blanks

df = pd.read_csv('moviereviews.tsv', sep='\t')

df_info(df)
check_categorical_columns(df)
remove_blanks(df)

print(f'The cleaned df has shape {df.shape}.')
