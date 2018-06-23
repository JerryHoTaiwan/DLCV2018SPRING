import torch
from transformer import Transformer
import itertools
from scipy import spatial
np.random.seed(2)

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
                    if 1-spatial.distance.cosine(class1_inner_vector[i, j], class2_inner_vector[k, l]) > 0:
                        quaruplets.append((i, j, k, l))
    return quaruplets

def get_transform_pair_seq(n_cluster):
    return list(itertools.permutations(range(n_cluster), 2))

def train(transformer, classifier, class1_centroids, class2_centroids, class1_label, n_epoch=50, lr=1e-4, batch_size=64):    
    transformer.train().cuda()
    classifier.eval()
    optimizer = torch.optim.SGD(transformer.parameters(), lr=lr)
    loss_cls = nn.CrossEntropyLoss()
    loss_mse = nn.MSELoss()

    n_cluster = len(class1_centroids)
    quaruplet_list = collect_quaruplets(class1_centroids, class2_centroids)
    class1_shuffled_centroids = np.random.shuffle(class1_centroids)
    class2_shuffled_centroids = np.random.shuffle(class2_centroids)

    for epoch in range(n_epoch):
        for i in range(0, len(quaruplet_list), batch_size):
            A, B, C, D = []
            for j in range(i, i+batch_size):
                A.append(torch.tensor(class1_centroids[quaruplet_list[j][0]]))
                B.append(torch.tensor(class1_centroids[quaruplet_list[j][1]]))
                C.append(torch.tensor(class1_centroids[quaruplet_list[j][2]]))
                D.append(torch.tensor(class1_centroids[quaruplet_list[j][3]]))
            A_tensor = torch.cat(A).cuda()
            B_tensor = torch.cat(B)
            C_tensor = torch.cat(C).cuda()
            D_tensor = torch.cat(D).cuda()

            B_hat = transformer(A_tensor, C_tensor, D_tensor)
            loss_mse_val = loss_mse(B_hat, B_tensor)
            class_prediction = classifier(B_hat)
            class_truth = np.empty(B_hat.shape[0]).fill(class1_label)
            loss_cls_val = loss_cls(class_prediction, class_truth)
            loss = loss_mse_val + wt*loss_cls_val

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('[epoch %d/%d] loss: %f, mse_loss: %f, cls_loss: %f'%(epoch+1, n_epoch, loss.item(), loss_mse_val.item(), loss_cls_val.item())
        torch.save({'epoch':epoch+1, 'state_dict':transformer.state_dict(), 'optimizer': optimizer.state_dict()}, 'transformer_%d.pth'%(epoch+1))

if __name__ == '__main__':
    n_cluster = 10
    feature_dim = 8192

    class1_centroids = np.random.rand(n_cluster, feature_dim)
    class2_centroids = np.random.rand(n_cluster, feature_dim)
    
    my_transformer = Transformer(feature_dim)
    # classifier = Classifier()