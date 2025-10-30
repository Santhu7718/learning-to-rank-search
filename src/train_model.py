import lightgbm as lgb
import pickle

def train_ranker(train_df, valid_df):
    X_train = train_df.drop(["relevance", "qid"], axis=1).values
    y_train = train_df['relevance'].values
    group_train = train_df.groupby('qid').size().values

    X_valid = valid_df.drop(["relevance", "qid"], axis=1).values
    y_valid = valid_df['relevance'].values
    group_valid = valid_df.groupby('qid').size().values

    train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, group=group_valid)

    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5, 10],
    }
    model = lgb.train(
        params, 
        train_data, 
        num_boost_round=100, 
        valid_sets=[train_data, valid_data],  
        callbacks=[lgb.early_stopping(stopping_rounds=10)]
    )

    # CORRECT indentation for saving model
    with open("model_fold1.pkl", "wb") as f:
        pickle.dump(model, f)

    return model
