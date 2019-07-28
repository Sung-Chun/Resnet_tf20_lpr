from __future__ import absolute_import, division, print_function
import tensorflow as tf
from models import resnet50, resnet101, resnet152, resnet34
import config
from prepare_data import get_datasets

# GPU settings
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# get the dataset
train_dataset, test_dataset, train_count, test_count = get_datasets()

# Use command tensorboard --logdir 'log' to start tensorboard
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='log')
callback_list = [tensorboard]

# start training
model = resnet50.ResNet50()
if config.network == "resnet34":
    model = resnet34.ResNet34()
if config.network == "resnet101":
    model = resnet101.ResNet101()
if config.network == "resnet152":
    model = resnet152.ResNet152()

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
              metrics=['accuracy'])

model.fit(train_dataset,
          epochs=config.EPOCHS,
          steps_per_epoch=train_count // config.BATCH_SIZE,
          validation_data=test_dataset,
          validation_steps=test_count // config.BATCH_SIZE,
          callbacks=callback_list)

# save the whole model
model.save(config.model_dir)
