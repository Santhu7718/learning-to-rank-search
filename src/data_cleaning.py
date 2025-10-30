def clean_letor_data(df):
    df = df.copy()
    feature_cols = [col for col in df.columns if col.startswith('feat_')]
    df[feature_cols] = df[feature_cols].replace([float('inf'), float('-inf')], float('nan'))
    df[feature_cols] = df[feature_cols].fillna(0.0)
    return df
