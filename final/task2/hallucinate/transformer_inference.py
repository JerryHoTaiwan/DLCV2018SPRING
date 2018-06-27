import torch
from transformer import Transformer
import scipy.misc as misc
import numpy as np
import itertools
import os
import random

n_cluster = 20
# def get_pair_from_clusters(centroids):
#     pair_list = []
#     for pair in itertools.permutations(range(1, 1+n_cluster), 2):
        
#     return np.array(pair_list)

if __name__ == '__main__':
    mode = 'train'
    submode = 'mean'
    checkpoint_path = 'model/transformer_78.pth'
    checkpoint_path_mean = 'model/transformer_31_mean.pth'
    
    classes_centroids_path = 'classes_centroids.npy'
    novel_classes_path = '../data/task2-dataset/novel/'
    ResNet_model_path = 'Res_trained_39.pkl'
    n_shot = 1
    feature_dim = 512
    n_base_class = 80
    n_novel_class = 20

    my_classifier = torch.load(ResNet_model_path)
    my_classifier.eval()

    novel_class_list = os.listdir(novel_classes_path)
    if mode == 'train':
        novel_imgs = []
        for novel_class_name in novel_class_list:
            for i in range(n_shot):
                img = misc.imread(novel_classes_path+novel_class_name+'/train/00%d.png'%(i+1))
                novel_imgs.append(img)
        novel_imgs_tensor = torch.Tensor(novel_imgs).view(-1, 3, 32, 32)
        # print(novel_imgs_tensor.shape)

        _, novel_features = my_classifier(novel_imgs_tensor.cuda())
        # print(novel_features.shape)
        if submode == 'mean':
            checkpoint = torch.load(checkpoint_path_mean)
        else:
            checkpoint = torch.load(checkpoint_path)
        transformer = Transformer(feature_dim)
        transformer.load_state_dict(checkpoint['state_dict'])
        transformer.eval().cuda()

        classes_centroids_np = np.load(classes_centroids_path)
        
        for i in range(n_novel_class):
            novel_classi_aug_feat_pool = []
            for novel_raw_feature in novel_features[i:i+n_shot]:
                novel_classi_aug_feat_pool.append(novel_raw_feature.cpu().detach().numpy())
                for classi in classes_centroids_np:
                    if submode == 'mean':
                        classi_centroid_mean = np.mean(classi, 0)
                        classi_shuffled = classi[:]
                        random.shuffle(classi_shuffled)
                        for classi_centroid in classi_shuffled[:5]:
                            novel_transformed = transformer(novel_raw_feature.cuda(), torch.tensor(classi_centroid_mean).cuda(), torch.tensor(classi_centroid).cuda()).cpu().detach().numpy()
                            novel_classi_aug_feat_pool.append(novel_transformed)
                    else:
                        pairs = itertools.permutations(classi, 2)
                        pairs_shuffled = pairs[:]
                        random.shuffle(pairs_shuffled)
                        for pair in pairs_shuffled[:5]:
                            novel_transformed = transformer(novel_raw_feature.cuda(), torch.tensor(pair[0]).cuda(), torch.tensor(pair[1]).cuda()).cpu().detach().numpy()
                            novel_classi_aug_feat_pool.append(novel_transformed)
            np.save('npy/'+novel_class_list[i]+'_mean.npy', np.array(novel_classi_aug_feat_pool))
            print(np.array(novel_classi_aug_feat_pool).shape)
    
    elif mode == 'val':
        for novel_class_name in novel_class_list:
            novel_imgs = []
            for i in range(100,500):
                img = misc.imread(novel_classes_path+novel_class_name+'/train/%03d.png'%(i+1))
                novel_imgs.append(img)
            novel_imgs_tensor = torch.Tensor(novel_imgs).view(-1, 3, 32, 32)
            _, novel_features = my_classifier(novel_imgs_tensor.cuda())
            print(novel_features.shape)
            np.save('npy/'+novel_class_name+'_val.npy', np.array(novel_features.cpu().detach()))