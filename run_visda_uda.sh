cd object/

python image_target.py --cls_par 0.3 --da uda --dset VISDA-C --gpu_id 0 --s 0 --output_src ckps/source/ --output ckps/target/ --net resnet101 --lr 1e-3 --max_epoch 2 --batch_size 32