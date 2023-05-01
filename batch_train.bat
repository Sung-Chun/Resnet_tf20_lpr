@echo off

python train.py -E=200 --width=130 --height=65 --model=resnet18 --savemodel-dir=resnet18_w130_65

echo %date%, %time%
