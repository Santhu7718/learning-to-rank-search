from sklearn.preprocessing import StandardScaler

def scale_features(df):
    feature_cols = [col for col in df.columns if col.startswith('feat_')]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df

def extract_group_sizes(df):
    return df.groupby('qid').size().to_list()
