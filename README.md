The Python evaluation tool for PDAF algorithms [Final version]
=================================
## Setup
* required external package: pillow, numpy, scipy, matplotlib, pandas, opencv-python (hope the version is fine)
* make sure all the .mat files are already existed in matfile/ 

## Usage

#### Preprocessing
* If you only want to evaluate the result of current DL methods, please skip the preprocessing
* If you have already generated new .flo files under new algorithm: python flo2npy.py  --data_path [the path of .flo files] --npy_name [the path you save .npy files] --down_sample [the scale of downsampling (default = 1)] ... for .flo files
* If you have already generated new .png files under new algorithm:python png2npy.py --data_path [the path of .flo files] --npy_name [the path you save .npy files] --down_sample [the scale of downsampling (default = 1)] ... for .png files
* If there are new scene for testing after 2018/08/16, please first modify line 46 in main.py. The signal already_set should be set as False.

the parameter 'down_sample' means the variance of scale between .flo files and .raw files. For example, if my .flo files are sized of 1024x384, this parameter should be set 2 since the size of raw file is 2048_768.<br>
For those who used .png files, make sure the calibration is correct in line 68 in png2npy.py. <br>

#### Evaluation
```
python main.py --save_path [the path to save the result] --test_data [the path of your testing .npy file] --method_index [the index of DL algorithms]
```
* if you doesn't have a new testing .npy file, please ignore it. The default .npy files is already setup
* For method_index, we set the default algorithm as '8_f2sd_bilinear__1024x384_resize' and '9_f2sd_bilinear__512x192_resize', which outperformed others. An example of modification: --method_index 0 1 2

#### Result
The following results will be generated in the saved path you set.
```
comparison/ ... The result of each metric under different scene and algorithm. The barchart is the weighted score of each algorithm.
disparity/ ... The estimation of different algorithms in the ROIs of each scene
error/ ... The error betweeen estimation and ground truth([1]) in the ROIs of each scene
iternum/ ... The nubmer of iteration in the ROIs of each scene.
txt files ... The process of autofocus in each ROI
```

#### The table of DL algorithms for comparison
```
0 11_f2sd_bilinear__1024x384_areaResize<br>
1 12_f2sd_bilinear_flow3__1024x384<br>
2 14_f2sd_2x_bilinear__1024x384<br>
3 101_f2sd_4x_bilinear_fq__512x192<br>
4 Set3_10bit_2016x756__f2sd__2048x768_shift<br>
5 Set3_10bit_2016x756__f2sd_bilinear__2048x768<br>
6 Set3_10bit_2016x756__f2sd_bilinear__2048x768_shift<br>
7 5_f2sd_bilinear__2048_768_zeroPad<br>
8 6_f2sd_bilinear__1984x704_cropFOV<br>
9 7_f2sd_bilinear__2048_768_resize<br>
10 8_f2sd_bilinear__1024x384_resize<br>
11 9_f2sd_bilinear__512x192_resize<br>
12 10_f2sd_4x_bilinear__2048x768_resize<br>
13 MTD_method_1<br>
14 MTD_method_2<br>
```

## Reference
An Efficient Auto Focus Method for Digital Still Camera Based on Focus Value Curve Prediction Model [[1]](https://pdfs.semanticscholar.org/06d2/ce627754c1d3d228a84467ab6844408e1cad.pdf)

## contact
Jerry Ho, NTUEE<br>
b04507009@ntu.edu.tw