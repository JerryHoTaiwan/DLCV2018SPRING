import numpy as np
import cv2
import glob
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNC

# As the description in P3-a, 30 is the number of key-point for each image.
category = ["Coast","Forest","Highway","Mountain","Suburb"]
color = ["red","orange","yellow","green","blue","pink"]
kp_num = 30

def collect(goal,Mat,count):
	for file in glob.glob(os.path.join(goal,'*')):
		img = cv2.imread(file,0)
		(kp,des) = get_kp(img)
		Mat[(count*30):][:30] = des
		count += 1

def get_kp(img):
	surf = cv2.xfeatures2d.SURF_create(50)
	kp,des = surf.detectAndCompute(img,None)
	kp = kp[:30]
	des = des[:30]
	return (kp,des)

def PCA(X, k=4):
    X_mean = np.mean(X,axis=0).astype(np.uint8)
    img_mean = X_mean.reshape(64,).astype(np.float64)

    X = X.T
    print (X.shape,X_mean.shape)
    X_mean = X_mean.reshape(64,1)
    X = X - np.repeat(X_mean,[X.shape[1]],axis=1)
    #do pca
    print("start doing svd ...")
    U, s, V = np.linalg.svd(X, full_matrices=False)
    print (X.shape,U.shape,s.shape,V.shape)
    s_sum = np.sum(s)
    for i in range(k):
    	print ('ratio of w_'+str(i)+': ',s[i]/s_sum)
    eigenfaces = U[:,:k]
    return eigenfaces,img_mean

def reconstruct(img,k,eigen,img_mean):
	f_img = img.flatten() - img_mean.flatten()
	weight = np.zeros(k)
	for i in range(k):
		weight[i] = np.dot(f_img,eigen[:,i])
	return weight

def create_table(FV,Km):
	cc = Km.cluster_centers_
	C_num = cc.shape[0]
	table = np.zeros((30,C_num))
	for k in range(30):
		for j in range(C_num):
			table[k][j] = np.linalg.norm(FV[k]-cc[j])
	return table

def HardSum(table):
	dim = table.shape[1]
	vec = np.zeros(dim)
	for i in range(30):
		agmin = np.argmin(table[i])
		vec[agmin] += 1
	return vec

def SoftSum(table):
	table_rec = 1 / table
	dim = table.shape[1]
	vec = np.zeros(dim)
	for i in range(30):
		vec += (table_rec[i] / np.sum(table_rec[i]))
	return vec

def SoftMax(table):
	dim = table.shape[1]
	vec = np.zeros(dim)
	table_rec = 1 / table
	table_tmp = np.zeros((30,dim))

	for i in range(30):
		table_tmp[i] = (table_rec[i] / np.sum(table_rec[i]))
	table_tmp = np.array(table_tmp)
	vec = np.amax(table_tmp,axis=0)
	return vec

def plot_3d(vec,path):
	fig,ax = plt.subplots(5,1)
	dim = vec.shape[1]
	print (vec.shape)
	x = np.arange(dim)
	for i in range(5):
		ax[i].bar(x,vec[i],color="blue")
		ax[i].set_title(category[i])
	fig.savefig("../result/" + path + "_10.png")

def P3_a():
	# part 3-a
	img_ran = cv2.imread("../Problem3/train-100/Highway/image_0009.jpg",0)
	(kp,des) = get_kp(img_ran)
	kp_ran = cv2.drawKeypoints(img_ran,kp,None,(0,0,255),4)
	cv2.imwrite("../result/kp_Highway_100.jpg",kp_ran)
	#img_test = np.pad(img_ran,24,mode='symmetric')
	#cv2.imwrite("../result/test_pad.jpg",img_test)

def P3_b():
	#part 3-b
	num = 100
	count = 0
	Mat_train = np.zeros((5*num*30,64)).astype(float)

	for item in category:
		path = "../Problem3/train-" + str(num) + "/" + item + "/"
		collect(path,Mat_train,count)
		count += num

	Km = KMeans(n_clusters=50,max_iter=5000).fit(Mat_train)
	cnt = 0
	for i in range(5*num*30):
		if (Km.predict(Mat_train)[i] < 6):
			cnt += 1

	FV = np.zeros((cnt,64))
	label = np.zeros(cnt)
	cnt_2 = 0
	for i in range(5*num*30):
		print (i)
		if (Km.predict(Mat_train)[i] < 6):
			label[cnt_2] = Km.predict(Mat_train)[i]
			FV[cnt_2] = Mat_train[i]
			cnt_2 += 1

	label = label.astype(int)
	eigen_dim,V_mean = PCA(FV,3)
	FV_3 = np.zeros((cnt,3))
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	for i in range(cnt_2):
		print (i)
		FV_3[i] = reconstruct(FV[i],3,eigen_dim,V_mean)
		ax.scatter(FV_3[i][0],FV_3[i][1],FV_3[i][2],c=color[label[i]])

	fig.savefig("../result/PCA_6clusters_100.png")

def P3_c():
	num = 10
	count = 0
	C = 50
	Mat_train = np.zeros((5*num*30,64)).astype(float)

	for item in category:
		path = "../Problem3/train-" + str(num) + "/" + item + "/"
		collect(path,Mat_train,count)
		count += num

	Km = KMeans(n_clusters=C,max_iter=1000).fit(Mat_train)

	table = np.zeros((5,30,C))
	HS = np.zeros((5,C))
	SS = np.zeros((5,C))
	SM = np.zeros((5,C))

	for i in range(5):
		table[i] = create_table(Mat_train[(i*num*30):][:30],Km)
		HS[i] = HardSum(table[i])
		SS[i] = SoftSum(table[i])
		SM[i] = SoftMax(table[i])

	plot_3d(HS,"HardSum")
	plot_3d(SS,"SoftSum")
	plot_3d(SM,"SoftMax")

def P3_d():
	num = 100
	test_num = 100
	count = 0
	C = 50
	label = np.array([0,1,2,3,4])
	label_train = np.repeat(label,num)
	Mat_train = np.zeros((5*num*30,64)).astype(float)

	for item in category:
		path = "../Problem3/train-" + str(num) + "/" + item + "/"
		collect(path,Mat_train,count)
		count += num

	Km = KMeans(n_clusters=C,max_iter=1000).fit(Mat_train)

	table = np.zeros((5*num,30,C))
	HS = np.zeros((5*num,C))
	SS = np.zeros((5*num,C))
	SM = np.zeros((5*num,C))

	for i in range(5*num):
		table[i] = create_table(Mat_train[(i*30):][:30],Km)
		HS[i] = HardSum(table[i])
		SS[i] = SoftSum(table[i])
		SM[i] = SoftMax(table[i])

# testing data
	count_test = 0
	Mat_test = np.zeros((5*test_num*30,64)).astype(float)
	label_test = np.repeat(label,test_num)
	table_test = np.zeros((5*test_num,30,C))
	HS_test = np.zeros((5*test_num,C))
	SS_test = np.zeros((5*test_num,C))
	SM_test = np.zeros((5*test_num,C))

	for test_item in category:
		path = "../Problem3/test-" + str(test_num) + "/" + test_item + "/"
		collect(path,Mat_test,count_test)
		count_test += test_num

	for i in range(5*test_num):
		table_test[i] = create_table(Mat_test[(i*30):][:30],Km)
		HS_test[i] = HardSum(table_test[i])
		SS_test[i] = SoftSum(table_test[i])
		SM_test[i] = SoftMax(table_test[i])

	print ("doing KNN...")
	nei = 11
	neigh_HS = KNC(n_neighbors=nei).fit(HS,label_train)
	print ("HardSum: " + str(neigh_HS.score(HS_test,label_test)))
	neigh_SS = KNC(n_neighbors=nei).fit(SS,label_train)
	print ("SoftSum: " + str(neigh_SS.score(SS_test,label_test)))
	neigh_SM = KNC(n_neighbors=nei).fit(SM,label_train)
	print ("SoftMax: " + str(neigh_SM.score(SM_test,label_test)))

def main():
	#P3_a()
	#P3_b()
	#P3_c()
	P3_d()

if __name__ == '__main__':
	main()