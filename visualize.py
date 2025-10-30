import matplotlib.pyplot as plt

folds = ['Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5']
ndcgs = [0.4878, 0.5081, 0.5260, 0.5775, 0.5705]

plt.plot(folds, ndcgs, marker='o')
plt.title('Test NDCG per Fold')
plt.xlabel('Fold')
plt.ylabel('NDCG')
plt.ylim(0, 1)
plt.grid(True)
plt.show()
