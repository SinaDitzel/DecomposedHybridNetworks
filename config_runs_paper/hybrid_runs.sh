#! /bin/bash
data_path="/data/GTSRB/"
input_size=48
ep=50
lr=1e-3
batch_size=64
num_workers=8
log_images=True
python=python

for i in {1..5}
do
    $python main.py --rgConf --lbpConf --input_size $input_size --data_path $data_path -ep $ep -lr $lr -bs $batch_size -nw $num_workers --log_images $log_images
    $python main.py --rg --lbp --input_size $input_size --data_path $data_path -ep $ep -lr $lr -bs $batch_size -nw $num_workers --log_images $log_images
    $python main.py --rgConf --lbp --input_size $input_size --data_path $data_path -ep $ep -lr $lr -bs $batch_size -nw $num_workers --log_images $log_images
    $python main.py --rg --lbpConf --input_size $input_size --data_path $data_path -ep $ep -lr $lr -bs $batch_size -nw $num_workers --log_images $log_images

    $python main.py --rgConf --lbpConf --reduced_classes --input_size $input_size --data_path $data_path -ep $ep -lr $lr -bs $batch_size -nw $num_workers --log_images $log_images
    $python main.py --rg --lbp --reduced_classes --input_size $input_size --data_path $data_path -ep $ep -lr $lr -bs $batch_size -nw $num_workers --log_images $log_images
    $python main.py --rgConf --lbp --reduced_classes --input_size $input_size --data_path $data_path -ep $ep -lr $lr -bs $batch_size -nw $num_workers --log_images $log_images
    $python main.py --rg --lbpConf --reduced_classes --input_size $input_size --data_path $data_path -ep $ep -lr $lr -bs $batch_size -nw $num_workers --log_images $log_images
done