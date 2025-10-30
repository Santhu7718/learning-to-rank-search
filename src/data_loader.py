import pandas as pd
import os

def parse_letor_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            tokens = line.strip().split()
            rel = int(tokens[0])
            qid = int(tokens[1].split(':')[1])
            features = []
            i = 2
            while i < len(tokens) and ':' in tokens[i]:
                features.append(float(tokens[i].split(':')[1]))
                i += 1
            data.append([rel, qid] + features)
    feature_cols = [f'feat_{i}' for i in range(1, len(features)+1)]
    cols = ['relevance', 'qid'] + feature_cols
    return pd.DataFrame(data, columns=cols)

def load_fold(fold_dir):
    train = parse_letor_file(os.path.join(fold_dir, 'train.txt'))
    test = parse_letor_file(os.path.join(fold_dir, 'test.txt'))
    vali = parse_letor_file(os.path.join(fold_dir, 'vali.txt'))
    return train, test, vali
