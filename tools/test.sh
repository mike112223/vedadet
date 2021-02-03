
NAME=$1

# for i in $(seq 1 6)  
# do
#     echo $i
#     python tools/test_nop.py \
#         configs/trainval/tad/${NAME}.py workdir/${NAME}/epoch_${i}00_weights.pth
# done  

for i in $(seq 7 12)  
do
    echo $i
    python tools/test_nop.py \
        configs/trainval/tad/${NAME}.py workdir/${NAME}/epoch_${i}00_weights.pth
done  
