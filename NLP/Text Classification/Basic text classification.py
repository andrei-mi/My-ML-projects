import numpy as np
import pandas as pd
from auxiliary_functions import check_categorical_columns, df_info

df = pd.read_csv('moviereviews.tsv', sep='\t')

df_info(df)
check_categorical_columns(df)
