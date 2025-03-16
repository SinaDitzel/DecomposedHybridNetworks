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
    $python main.py --reconstruction --rgConf --lbpConf --input_size $input_size --data_path $data_path -ep $ep -lr $lr -bs $batch_size -nw $num_workers --log_images $log_images
    $python main.py --reconstruction --rg --lbp --input_size $input_size --data_path $data_path -ep $ep -lr $lr -bs $batch_size -nw $num_workers --log_images $log_images
    $python main.py --reconstruction --rgConf --lbp --input_size $input_size --data_path $data_path -ep $ep -lr $lr -bs $batch_size -nw $num_workers --log_images $log_images
    $python main.py --reconstruction --rg --lbpConf --input_size $input_size --data_path $data_path -ep $ep -lr $lr -bs $batch_size -nw $num_workers --log_images $log_images

    $python main.py --reconstruction --reduced_classes --rgConf --lbpConf --input_size $input_size --data_path $data_path -ep $ep -lr $lr -bs $batch_size -nw $num_workers --log_images $log_images
    $python main.py --reconstruction --reduced_classes --rg --lbp --input_size $input_size --data_path $data_path -ep $ep -lr $lr -bs $batch_size -nw $num_workers --log_images $log_images
    $python main.py --reconstruction --reduced_classes --rgConf --lbp --input_size $input_size --data_path $data_path -ep $ep -lr $lr -bs $batch_size -nw $num_workers --log_images $log_images
    $python main.py --reconstruction --reduced_classes --rg --lbpConf --input_size $input_size --data_path $data_path -ep $ep -lr $lr -bs $batch_size -nw $num_workers --log_images $log_images

done