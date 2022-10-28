# TensorFlow2.0_ResNet
A ResNet(**ResNet18, ResNet34, ResNet50, ResNet101, ResNet152**) implementation using TensorFlow-2.0

이 github는 https://github.com/calmisential/TensorFlow2.0_ResNet 를 기반으로 하고 있습니다.

## 개발 및 테스트 환경
+ Python 3.9
+ Tensorflow 2.10.0

## Train 하기
1. 데이터셋.zip 파일에서 image 폴더 아래의 1996_n, 2004_n, ..., echo 의 폴더 및 그 아래 이미지들을 **original dataset** 폴더 아래에 푼다. 아래와 같은 폴더 구조가 될 것이다.:
```
|——original dataset
   |——1996_n
   |——2004_n
   |——2006_eu
   |——2006_n
```
2. 여기서 라벨 구분을 용이하게 하기 위해, 각 폴더명 앞에 01, 02, 03, ... 을 붙여준다. 그러면 아래와 같은 폴더명이 된다.:
```
|——original dataset
   |——00_1996_n
   |——01_2004_n
   |——02_2006_eu
   |——03_2006_n
```

3. Run the script **split_dataset.py** to split the raw dataset into train set, valid set and test set.
python split_dataset.py

4. Run **train.py** to start training. \
python train.py -h \
usage: train.py [-h] -E EPOCH [-B BATCH_SIZE] [--width WIDTH] [--height HEIGHT] [--model MODEL] --savemodel-dir \
                SAVEMODEL_DIR [--partition-name PARTITION_NAME] [--resource-orientation RESOURCE_ORIENTATION]

예) python train.py -E=3000 --model=resnet50 --savemodel-dir=save_resnet50

    Resnet50 모델을 사용하여 Epoch=3000까지 훈련하고, weight를 save_resnet50 폴더에 저장한다. 1000 epoch 단위로 중간 weight를 저장한다.

## Evaluate
Run **evaluate.py** to evaluate the model's performance on the test dataset.

## Inference
예) python predict.py --epoch=2000 --model=resnet50 --savemodel-dir=save_resnet50

    최종 epoch까지 훈련된 모델을 사용하여 prediction을 하면 --epoch=final 이라고 지정하면 되고,
    중간 weight를 사용하려면 (예를 들어 epoch=2000에 중간 저장된 weight를 사용하려면) --epoch=2000 이라고 지정한다.

