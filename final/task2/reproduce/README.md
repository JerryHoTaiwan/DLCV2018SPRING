# Task 2

## Requirements
  * Python 3.6.4
  * Torch 0.4.0
  * torchvision 0.2.0
  * matplotlib 2.2.2
  * opencv 3.4

## Usage
   
   ```
      python3 test_siamese.py [the path of novel classes] [the path of testing data] [number of shots]
   ```
   
   Example:
  
   ```
       python3 test_siamese.py ../task2-dataset/novel/ ../test/ 5
   ``` 
   
   Note that here we assume the folders are exactly same as which we downloaded. 
   
   That is, each novel classes will include a folder named 'train' which includes images. 
   
   Ex: task2-data/novel/class_00/train/689.png

   After the execution, a `.csv` file of prediction will be generated, which would be names as `predict_k.csv` for k shot problem.


## Result
    
   * the score of our prediction on kaggle   
        
        
        k-shot    | accuracy  |
        --------- | ----------
        1         | 0.34700 
        5         | 0.51600
        10        | 0.53200
        
