CUDA_VISIBLE_DEVICES=4 python getting_embeddings.py \
  --input ../Genearted_sequences/val.fasta \
  --output ../Generated_representations/val_esm2_8m.npz \
  --pooling mean max min cls layer_mean \
  --batch-size 128