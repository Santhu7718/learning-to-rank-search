import os
from src.data_loader import load_fold
from src.data_cleaning import clean_letor_data
from src.feature_engineering import scale_features
from src.train_model import train_ranker
from src.evaluate import compute_ndcg

DATA_ROOT = "dataset/MQ2008/"
all_results = []

for fold in ["Fold1", "Fold2", "Fold3", "Fold4", "Fold5"]:
    print(f"\nProcessing {fold} ...")
    train, test, valid = load_fold(os.path.join(DATA_ROOT, fold))
    train = clean_letor_data(train)
    test = clean_letor_data(test)
    valid = clean_letor_data(valid)

    # Optional: Feature scaling
    train = scale_features(train)
    test = scale_features(test)
    valid = scale_features(valid)

    # Train model
    model = train_ranker(train, valid)

    # Predict
    X_test = test.drop(["relevance", "qid"], axis=1).values
    preds = model.predict(X_test)

    # Evaluate
    ndcg = compute_ndcg(preds, test)
    print(f"{fold}: Test NDCG = {ndcg:.4f}")
    all_results.append(ndcg)

print(f"\nAverage NDCG across folds: {sum(all_results)/len(all_results):.4f}")
