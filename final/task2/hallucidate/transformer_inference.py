import torch
from transformer import Transformer
import scipy.misc as misc

if __name__ == '__main__':
    checkpoint_path = ''
    class2_centroids_path = ''
    n_shot = 1
    feature_dim = 8192

    checkpoint = torch.load(checkpoint_path)
    transformer = Transformer(feature_dim)
    transformer.load_state_dict(checkpoint['state_dict'])
    transformer.eval().cuda()

    class2_centroids = np.load(class2_centroids_path)
