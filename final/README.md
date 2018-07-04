# Final Project

<!-- /code_chunk_output -->

## Tasks
  * Task 1: Small Data Supervised Learning
    * Train classifier using a limited version of Fashion MNIST dataset 
  * Task 2: One-shot / Few-shot Learning
    * Design model to recognize a number of novel classes with insufficient number of training images

## Requirements
  * Python 3.6.4
  * Torch 0.4.0
  * torchvision 0.2.0
  * scipy 1.0.1
  * matplotlib 2.2.2
  * numpy 1.14.2


## Usage
    
   * Task 1:
      
      First, get into the root file for task 1, `cd task1`
      
      Put the file `Fashion_MNIST_student` into the directory `datasets` as shown
      
      * For training, please check the argument listed in `train.py`
        
        ```
            python3 train.py --train-dir datasets/Fashion_MNIST_student/train
                             --test-dir datasets/Fashion_MNIST_student/test
                             --batch-size 128
                             --epochs 100
                             --save-freq 1
        ```       
           
      * For testing, please check the argument listed in `inference.py`
      
        ```
            python3 inference.py --train-dir datasets/Fashion_MNIST_student/train
                                 --test-dir datasets/Fashion_MNIST_student/test
                                 --checkpoint checkpoints/fashion_mnist/best_checkpoint.pth.tar
        ``` 
      
      * For Kaggle results, execute the script `task1.sh` for download model and inference the result
        
        ```
            bash task1.sh <train directory> <test directory>
        ```
        
        Example: the path is same as listed above.
        ```
            bash task1.sh datasets/Fashion_MNIST_student/train datasets/Fashion_MNIST_student/test
        ```
        
        After execution, there will be a file called `result.csv`
   
   * Task 2:
        * Please access to task2/reproduce for detailed information ([link](https://github.com/chinchengwu/DLCV2018SPRING/tree/master/final/task2/reproduce))
