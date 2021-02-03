
NAME=$1

for i in $(seq 1 6)  
do
    echo $i
    python tools/test_nop.py \
        configs/trainval/tad/${NAME}.py /DATA_DB25/sdc/data/yanjiazhu/code/vedadet_workdir/${NAME}/epoch_${i}00_weights.pth
done  
