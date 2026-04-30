CUDA_VISIBLE_DEVICES=6 python getting_embeddings.py \
  --input ../Genearted_sequences/train.fasta \
  --output ../Generated_representations/train_esm2_8m.npz \
  --pooling mean max min cls layer_mean \
  --batch-size 128