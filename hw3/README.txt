For the test file please input the command:

bash hw3.sh <testing image directory> <output images directory>
bash hw3_best.sh <testing image directory> <output images directory>

If there were problems, please execute the training file with the following command

python3 parse_data.py <training image directory> <validation images directory>

You will get .npy files for training. Please execute the file hw3_train.py next.

python3 hw3_train.py <model directory> <mode>

the mode can be 32,16,or 8.