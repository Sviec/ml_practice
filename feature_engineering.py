from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def feature_engineering(df):
    df['BsmtUsability'] = df['BsmtQual'] + "_" + df['BsmtExposure']
    df['ExterScore'] = df['ExterQual'] + "_" + df['ExterCond']
    df['BsmtFinTypeScore'] = df['BsmtFinType1'] + "_" + df['BsmtFinType2']
    df['BsmtBathTotal'] = df['BsmtFullBath'] + df['BsmtHalfBath']
    df['BathTotal'] = df['FullBath'] + df['HalfBath']
    df['OverallProduct'] = df['OverallQual'] * df['OverallCond']
    df = df.drop(
        columns=['BsmtQual', 'BsmtExposure', 'ExterQual', 'ExterCond',
                 'BsmtFinType1', 'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath',
                 'FullBath', 'HalfBath', 'OverallQual', 'OverallCond'],
        axis=1
    )

    encoder = LabelEncoder()

    for feature in df.select_dtypes(include=['object']).columns:
        df[feature] = encoder.fit_transform(df[feature])
    return df
