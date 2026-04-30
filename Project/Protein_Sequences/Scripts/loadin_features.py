import numpy as np

train_data = np.load(
    "../fitted_features/embeddings_mean_50/embeddings_mean_pca_50_train_features.npz",
    allow_pickle=True,
)

val_data = np.load(
    "../val_features/embeddings_mean_50/embeddings_mean_pca_50_val_features.npz",
    allow_pickle=True,
)

test_data = np.load(
    "../test_features/embeddings_mean_50/embeddings_mean_pca_50_test_features.npz",
    allow_pickle=True,
)

X_train = train_data["features"]
X_val = val_data["features"]
X_test = test_data["features"]

train_ids = train_data["ids"]
val_ids = val_data["ids"]
test_ids = test_data["ids"]

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)