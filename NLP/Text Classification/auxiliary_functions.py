def df_info(df):
    print(f'The shape of the dataframe is {df.shape}')
    print('\n-----------------------------------------\n')
    print(f'The types of the columns are:\n\n{df.dtypes}')
    print('\n-----------------------------------------\n')
    print(f'The number of NANs in each column is:\n\n{df.isnull().sum()}')
    print('\n-----------------------------------------\n')
    df_dropna = df.dropna()
    if len(df_dropna)/len(df) > 0.8:
        df.dropna(inplace=True)
    print(f'The shape of the dataframe after dropping the NANs(if necessary) is {df.shape}')
    print('\n-----------------------------------------\n')


def check_categorical_columns(df):
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    for col in categorical_columns:
        entirely_empty_strings_count = (df[col] == '').sum()
        space_string_count = df[col].str.isspace().sum()
        print(f'Column {col} has {df[col].isnull().sum()} nulls \
              and {entirely_empty_strings_count} entirely empty strings \
              and {space_string_count} only space strings')
        print('\n')
