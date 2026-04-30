python apply_feature_extractors.py \
  --input ../Generated_representations/test_esm2_8m.npz \
  --embedding-key embeddings_mean \
  --fitted-dir ../fitted_features/embeddings_mean_50 \
  --output-dir ../test_features/embeddings_mean_50 \
  --split-name test