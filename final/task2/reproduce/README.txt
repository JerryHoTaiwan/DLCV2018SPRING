Group 14 task 2

Required package: OpenCV 3.4 (only for imread, the version may be fine), Pytorch, torchvision

Usage: python3 test_siamese.py [the path of novel classes] [the path of testing data] [number of shots]

For example: python3 test_siamese.py ../task2-dataset/novel/ ../test/ 5
(Note that here we assume the folders are exactly same as which we downloaded. That is, each novel classes will include a folder named 'train' which includes images. Ex: task2-data/novel/class_00/train/689.png)

After the execution, a .csv file of prediction will be generated, which would be names as predict_k.csv for k shot problem.

the score of our prediction on kaggle
1-shot: 0.34700
5-shot: 0.51600
10-shot: 0.53200

Hope we can have a happy summer vacation after this project lol