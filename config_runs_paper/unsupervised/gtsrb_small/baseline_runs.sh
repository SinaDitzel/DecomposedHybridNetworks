#! /bin/bash
data_path="/data/GTSRB/"
input_size=128
ep=50
lr=1e-3
batch_size=64
num_workers=8
log_images=False
python=python

for i in {1..5}
do
    $python main.py --reduced_classes --input_size $input_size -unsup --data_path $data_path -ep $ep -lr $lr -bs $batch_size -nw $num_workers --log_images $log_images
done