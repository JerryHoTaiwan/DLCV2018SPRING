import torch
from transformer import Transformer
import itertools
from scipy import spatial
import ResNetFeat
import numpy as np
import torch.nn as nn
np.random.seed(2)

def collect_mean_quaruplets(class1_centroids, class2_centroids):
    n_cluster = class1_centroids.shape[0]
    feature_dim = class1_centroids.shape[1]

    class1_mean = np.mean(class1_centroids, 0)
    class2_mean = np.mean(class2_centroids, 0)
    print(class2_mean.shape)

    class1_vectors = np.zeros((n_cluster, feature_dim))
    class2_vectors = np.zeros((n_cluster, feature_dim))

    for i in range(n_cluster):
        class1_vectors[i] = class1_centroids[i]-class1_mean
        class2_vectors[i] = class2_centroids[i]-class2_mean

    quaruplets = []
    for i in range(n_cluster):
        for j in range(n_cluster):
            if 1-spatial.distance.cosine(class1_vectors[i], class2_vectors[j]) > 0:
                quaruplets.append((i, j))
    return quaruplets

def collect_quaruplets(class1_centroids, class2_centroids):
    n_cluster = class1_centroids.shape[0]
    feature_dim = class1_centroids.shape[1]

    class1_inner_vector = np.zeros((n_cluster, n_cluster, feature_dim))
    class2_inner_vector = np.zeros((n_cluster, n_cluster, feature_dim))
    
    for i in range(n_cluster):
        for j in range(n_cluster):
            class1_inner_vector[i, j] = class1_centroids[i]-class1_centroids[j]
            class2_inner_vector[i, j] = class2_centroids[i]-class2_centroids[j]

    quaruplets = []
    for i in range(n_cluster):
        for j in range(n_cluster):
            for k in range(n_cluster):
                for l in range(n_cluster):
                    if 1-spatial.distance.cosine(class1_inner_vector[i, j], class2_inner_vector[k, l]) > 0.5:
                        quaruplets.append((i, j, k, l))
    return quaruplets

# this function is not usedin current version
def get_transform_pair_seq(n_cluster):
    return list(itertools.permutations(range(n_cluster), 2))

def train(transformer, classifier, optimizer, class1_centroids, class2_centroids, class1_label, n_epoch=20, batch_size=64, wt=10, save_model_path='model/'):
    mode = 'mean'
    transformer.train().cuda()
    classifier.eval()
    loss_cls = nn.CrossEntropyLoss()
    loss_mse = nn.MSELoss()

    n_cluster = len(class1_centroids)
    if mode == 'mean':
        quaruplet_list = collect_mean_quaruplets(class1_centroids, class2_centroids)
        class1_mean = np.mean(class1_centroids, 0)
        class2_mean = np.mean(class2_centroids, 0)

        for epoch in range(n_epoch):
            for i in range(0, len(quaruplet_list), batch_size):
                A, B, C, D = [], [], [], []
                for j in range(i, min(i+batch_size, len(quaruplet_list))):
                    A.append(torch.tensor(class1_mean))
                    B.append(torch.tensor(class1_centroids[quaruplet_list[j][0]]))
                    C.append(torch.tensor(class2_mean))
                    D.append(torch.tensor(class2_centroids[quaruplet_list[j][1]]))
                A_tensor = torch.stack(A).cuda()
                B_tensor = torch.stack(B).cuda()
                C_tensor = torch.stack(C).cuda()
                D_tensor = torch.stack(D).cuda()
                # print(A_tensor.shape)

                B_hat = transformer(A_tensor, C_tensor, D_tensor)
                loss_mse_val = loss_mse(B_hat, B_tensor).cpu()
                class_prediction = classifier.classifier(B_hat).cpu()
                class_truth = torch.empty(B_hat.shape[0], dtype=torch.long).fill_(class1_label).cpu()
                loss_cls_val = loss_cls(class_prediction, class_truth)
                loss = loss_mse_val + wt*loss_cls_val

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print('[epoch %d/%d] loss: %f, mse_loss: %f, cls_loss: %f'%(epoch+1, n_epoch, loss.item(), loss_mse_val.item(), loss_cls_val.item()))
        torch.save({'state_dict':transformer.state_dict(), 'optimizer': optimizer.state_dict()}, save_model_path+'transformer_%d_mean.pth'%(class1_label))

    else:
        quaruplet_list = collect_quaruplets(class1_centroids, class2_centroids)

        for epoch in range(n_epoch):
            for i in range(0, len(quaruplet_list), batch_size):
                A, B, C, D = [], [], [], []
                for j in range(i, min(i+batch_size, len(quaruplet_list))):
                    A.append(torch.tensor(class1_centroids[quaruplet_list[j][0]]))
                    B.append(torch.tensor(class1_centroids[quaruplet_list[j][1]]))
                    C.append(torch.tensor(class2_centroids[quaruplet_list[j][2]]))
                    D.append(torch.tensor(class2_centroids[quaruplet_list[j][3]]))
                A_tensor = torch.stack(A).cuda()
                B_tensor = torch.stack(B).cuda()
                C_tensor = torch.stack(C).cuda()
                D_tensor = torch.stack(D).cuda()
                # print(A_tensor.shape)

                B_hat = transformer(A_tensor, C_tensor, D_tensor)
                loss_mse_val = loss_mse(B_hat, B_tensor).cpu()
                class_prediction = classifier.classifier(B_hat).cpu()
                class_truth = torch.empty(B_hat.shape[0], dtype=torch.long).fill_(class1_label).cpu()
                loss_cls_val = loss_cls(class_prediction, class_truth)
                loss = loss_mse_val + wt*loss_cls_val

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print('[epoch %d/%d] loss: %f, mse_loss: %f, cls_loss: %f'%(epoch+1, n_epoch, loss.item(), loss_mse_val.item(), loss_cls_val.item()))
        torch.save({'state_dict':transformer.state_dict(), 'optimizer': optimizer.state_dict()}, save_model_path+'transformer_%d.pth'%(class1_label))

if __name__ == '__main__':
    feature_dim = 512
    n_classes = 80
    
    classes_centroids_npy = 'classes_centroids.npy'
    classes_centroids = np.load(classes_centroids_npy)
    ResNet_model_path = 'Res_trained_39.pkl'
    
    my_transformer = Transformer(feature_dim)
    # my_classifier = ResNetFeat.ResNet18(num_classes=80)
    # my_classifier.load_state_dict(torch.load(ResNet_model_path))
    my_classifier = torch.load(ResNet_model_path)
    optimizer = torch.optim.Adam(my_transformer.parameters(), lr=1e-4)
    # print(my_classifier)

    for i in range(n_classes):
        for j in range(i+1, n_classes):
            class1_centroids = classes_centroids[i]
            class2_centroids = classes_centroids[j]
            train(my_transformer, my_classifier, optimizer, class1_centroids, class2_centroids, i)