Feature Extractor by ResNet

-- readfile.py: To generate the .npy files with normalization. Note that u have to modify the path.

-- ResNetBasic.py: The basic structure of ResNet, five types in all.

-- ResNetFeat.py: Output both feature vectors and predictive score.

-- train.py: I used all the parameteres as default values here lol

-- read_features.py: Usage: python3 read_feature.py [the path of .npy folder] [the folder to save the concatenated .npy files and labels] [the number u samples for each base lasses for training] [the number u samples for each base lasses for validation]

-- train_f2c.py: Usage: python3 train_f2c.py [the path to load training file] [the path to save your model]

6/27