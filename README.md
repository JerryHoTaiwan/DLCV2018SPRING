The Python evaluation tool for PDAF algorithms [Final version]
=================================
## Setup
required external package: pillow, numpy, scipy, matplotlib, pandas, opencv-python (hope the version is fine)
image size: 

## Usage

#### Preprocessing
If the user only want to evaluate the result of current DL methods, please skip the preprocessing

#### The list of DL algorithms for comparison
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
Jerry<br>
contact: b04507009@ntu.edu.tw