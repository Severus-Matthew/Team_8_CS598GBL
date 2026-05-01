import numpy as np

val_data = np.load(
    "../val_features/embeddings_mean_32/embeddings_mean_pca_32_val_features.npz",
    allow_pickle=True,
)

test_data = np.load(
    "../test_features/embeddings_mean_32/embeddings_mean_pca_32_test_features.npz",
    allow_pickle=True,
)

X_val = val_data["features"]
X_test = test_data["features"]

val_ids = val_data["ids"]
test_ids = test_data["ids"]

print(X_val.shape)
print(X_test.shape)