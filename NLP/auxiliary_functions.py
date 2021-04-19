from sklearn.metrics import roc_auc_score


def df_info(df):
    print(f'The initial shape of the dataframe is {df.shape}.')
    print('\n-----------------------------------------\n')
    print(f'The types of the columns are:\n\n{df.dtypes}')
    print('\n-----------------------------------------\n')
    print(f'The number of NANs in each column is:\n\n{df.isnull().sum()}')
    print('\n-----------------------------------------\n')
    df_dropna = df.dropna()
    if len(df_dropna)/len(df) > 0.8:
        df.dropna(inplace=True)
    print(f'The shape of the dataframe after dropping the NANs(if necessary) is {df.shape}.')
    print('\n-----------------------------------------\n')


def check_categorical_columns(df):
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    print('Info about empty/space strings in each categorical column:\n\n')
    for col in categorical_columns:
        entirely_empty_strings_count = (df[col] == '').sum()
        space_string_count = df[col].str.isspace().sum()
        print(f'Column {col} has {entirely_empty_strings_count} entirely empty strings and {space_string_count} only space strings')
    print('\n-----------------------------------------\n')


def remove_blanks(df):
    blanks = []
    for i, lb, rv in df.itertuples():
        if rv.isspace():
            blanks.append(i)
    df.drop(blanks, inplace=True)
    return df


def map_label(label):
    if label == 'pos':
        return 1
    else:
        return 0


def processed_auroc(y_test, predictions):
    mapped_y_test = list(map(map_label, y_test))
    mapped_predictions = list(map(map_label, predictions))
    return roc_auc_score(mapped_y_test, mapped_predictions)
