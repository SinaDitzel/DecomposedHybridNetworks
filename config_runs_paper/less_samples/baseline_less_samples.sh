#! /bin/bash
data_path="/data/GTSRB/"
input_size=48
lr=1e-3
batch_size=64
num_workers=8
log_images=False
python=python

for seed in 0 1 2 3 4 
do
    for i in 100,210 50,465 10,2326 5,4651 #num_samples + epochs(higher to keep same iterations) 
    do
        IFS=","
        set -- $i # convert the "tuple" into the param args $1 $2
        $python main.py -p $1 -ep $2 --seed $seed --input_size $input_size --data_path $data_path -lr $lr -bs $batch_size -nw $num_workers --log_images $log_images
        $python main.py -p $1 -ep $2 --seed $seed --reduced_classes --input_size $input_size --data_path $data_path -lr $lr -bs $batch_size -nw $num_workers --log_images $log_images
    done
done


