Representation and Feature Analysis in Protein Language Models

Feature Paths: `Project/Protein_Sequences/{split}_features/embeddings_{pooling-method}_{number-of-features}/embeddings_{pooling-method}_{extraction-method}_{number-of-features}_{split}_features.nps`
where 
- `split = {val, test}`
- `pooling-method = {mean, max, min, cls, layer_mean}`
- `number-of-features = {8, 16, 32, 64, 128}`
- `extraction-method = {pca, ica, nmf}`

See `Project/Protein_Sequences/Scripts/loadin_features.py` for an example of how to read in the feature files.