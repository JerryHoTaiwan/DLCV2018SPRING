#!/bin/bash
wget -O 'VAE_97_model2.pkl' 'https://www.dropbox.com/s/39bk0n4ufhrtu18/VAE_97_model2.pkl?dl=1'
wget -O 'GAN_Generator.pkl' 'https://www.dropbox.com/s/6pqqkgfraadozgw/GAN_Generator.pkl?dl=1'
wget -O 'ACGAN_G_199.pkl' 'https://www.dropbox.com/s/p6hwrs1qcnhzf5n/ACGAN_G_199.pkl?dl=1'

python3 GAN_test.py $1 $2
python3 ACGAN_test.py $1 $2
python3 VAE_test.py $1 $2
