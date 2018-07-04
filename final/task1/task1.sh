wget -O best_checkpoint.pth.tar https://www.dropbox.com/s/vaxm7ed558vfl71/best_checkpoint.pth.tar?dl=1
python3 inference.py --checkpoint best_checkpoint.pth.tar --train-dir $1 --test-dir $2
