from sklearn.impute import SimpleImputer


def preprocess(df):
    drop = df.columns[(df.isna().sum() / df.shape[0]) > 0.3]

    df = df.drop(drop, axis=1)

    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    imputer_median = SimpleImputer(strategy='median')
    df[numerical_cols] = imputer_median.fit_transform(df[numerical_cols])

    imputer_most = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = imputer_most.fit_transform(df[categorical_cols])

    return df