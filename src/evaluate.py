from sklearn.metrics import ndcg_score

def compute_ndcg(preds, df):
    qids = df['qid'].unique()
    ndcgs = []
    for qid in qids:
        idx = df['qid'] == qid
        true = df.loc[idx, 'relevance'].values.reshape(1, -1)
        pred = preds[idx].reshape(1, -1)
        ndcgs.append(ndcg_score(true, pred))
    return sum(ndcgs) / len(ndcgs)
