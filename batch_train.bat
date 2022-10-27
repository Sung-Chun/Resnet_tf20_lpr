@echo off

python train.py -E=10000 --width=224 --height=128 --model=resnet50 --savemodel-dir=resnet50_w224_h128
python train.py -E=10000 --width=224 --height=128 --model=resnet101 --savemodel-dir=resnet101_w224_h128
python train.py -E=10000 --width=224 --height=128 --model=resnet152 --savemodel-dir=resnet152_w224_h128

echo %date%, %time%
