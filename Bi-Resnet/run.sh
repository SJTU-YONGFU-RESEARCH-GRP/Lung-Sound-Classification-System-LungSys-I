#!/bin/sh 

#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}/usr/local/cuda/extras/CUPTI/lib64
#sudo sh run.sh
#ps -aux | grep "resnet.py"

#ls log/outfile||sudo mkdir log/outfile &
#export CUDA_VISIBLE_DEVICES="2"
# for varible in {0..1}; do
    # ls log/V3.24.$varible||mkdir log/V3.24.$varible &
# done
	
CUDA_VISIBLE_DEVICES=1 python3 model/bnn.py \
--lr 0.0001 \
--save log/report/V3.24.0 \
--gpu 0 \
--nepochs 200 \
--input ./analysis/pack/wavelet_stft_train.p \
--test ./analysis/pack/wavelet_stft_test.p \
--batch_size 32 \
--weight_decay 0 \
--nonLocal True \
--comment V3.24.0 \
> log/outfile/myout3_24_0.file 2>&1&
  
#CUDA_VISIBLE_DEVICES=2 python3 model/bnn.py --lr 0.0001 --save log/report/V3.24.1 --gpu 0 --nepochs 200 --input ./analysis/pack/wavelet_stft_train.p --test ./analysis/pack/wavelet_stft_test.p --batch_size 128 --weight_decay 0 --nonLocal False --comment V3.24.1 > log/outfile/myout3_24_1.file 2>&1&