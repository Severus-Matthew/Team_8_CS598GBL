for n in 8 16 32 64 128; do
for method in mean max min cls layer_mean; do
python fit_feature_extractors.py \
  --input ../Generated_representations/train_esm2_8m.npz \
  --embedding-key embeddings_$method \
  --methods pca ica nmf \
  --n-components $n \
  --output-dir ../fitted_features/embeddings_$method_$n
done
done