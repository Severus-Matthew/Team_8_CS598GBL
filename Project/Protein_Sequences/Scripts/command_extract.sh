python fit_feature_extractors.py \
  --input ../Generated_representations/val_esm2_8m.npz \
  --embedding-key embeddings_mean \
  --methods pca ica nmf \
  --n-components 50 \
  --output-dir ../fitted_features/embeddings_mean_50
