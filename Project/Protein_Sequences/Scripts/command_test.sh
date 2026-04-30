CUDA_VISIBLE_DEVICES=5 python getting_embeddings.py \
  --input ../Genearted_sequences/test.fasta \
  --output ../Generated_representations/test_esm2_8m.npz \
  --pooling mean max min cls layer_mean \
  --batch-size 128