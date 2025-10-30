def get_features_targets_groups(df):
    X = df.drop(['relevance', 'qid'], axis=1).values
    y = df['relevance'].values
    groups = df.groupby('qid').size().values
    return X, y, groups
