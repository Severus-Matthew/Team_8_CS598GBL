for method in mean max min cls layer_mean; do
  for n in 8 16 32 64 128; do
  for split in val test; do
  python apply_feature_extractors.py \
    --input ../Generated_representations/${split}_esm2_8m.npz \
    --embedding-key embeddings_$method \
    --fitted-dir ../fitted_features/embeddings_${method}_${n} \
    --output-dir ../${split}_features/embeddings_${method}_${n} \
    --split-name $split
    done
  done
done