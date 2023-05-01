from __future__ import absolute_import, division, print_function
import tensorflow as tf
from models.resnet import resnet_18, resnet_34, resnet_50, resnet_101, resnet_152
from prepare_data import generate_datasets, generate_corner_datasets
import math

import os, sys
import time, datetime
import argparse

def get_model(model_name, image_width, image_height, channels, regressor=False):
    if model_name == "resnet18":
        model = resnet_18(regressor)
    elif model_name == "resnet34":
        model = resnet_34(regressor)
    elif model_name == "resnet50":
        model = resnet_50(regressor)
    elif model_name == "resnet101":
        model = resnet_101(regressor)
    elif model_name == "resnet152":
        model = resnet_152(regressor)
    else:
        return None

    model.build(input_shape=(None, image_height, image_width, channels))
    model.summary()
    return model

def get_argparser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-E', '--epoch', type=int, required=True)
    parser.add_argument('-B', '--batch-size', type=int, default=8)
    parser.add_argument('--width', type=int, default=224, help='image width')
    parser.add_argument('--height', type=int, default=128, help='image height')
    parser.add_argument('--model', type=str, default='resnet50', help='resnet model name')
    parser.add_argument('--regression', type=bool, default=False, help='resnet classifier/regressor')
    parser.add_argument('--savemodel-dir', type=str, required=True, help='directory to save model')

    parser.add_argument('--partition-name', type=str, required=False)
    parser.add_argument('--resource-orientation', type=str, required=False)

    return parser

if __name__ == '__main__':

    # Argument parsing
    parser = get_argparser()
    args = parser.parse_args()

    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


    # start time (checking time to take for training)
    start_t = time.time()

    # create model
    model = get_model(args.model, args.width, args.height, channels=1, regressor=args.regression)
    if model is None:
        print(f'--model parameter is wrong: {args.model}')
        sys.exit(0)


    if args.regression:
        # get the original_dataset
        train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_corner_datasets(args.batch_size, args.width, args.height)

        # define loss and optimizer
        loss_object = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adadelta()

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.MeanSquaredError(name='train_accuracy')

        valid_loss = tf.keras.metrics.Mean(name='valid_loss')
        valid_accuracy = tf.keras.metrics.MeanSquaredError(name='valid_accuracy')
    else:
        # get the original_dataset
        train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets(args.batch_size, args.width, args.height)

        # define loss and optimizer
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adadelta()

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        valid_loss = tf.keras.metrics.Mean(name='valid_loss')
        valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(y_true=labels, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def valid_step(images, labels):
        predictions = model(images, training=False)
        v_loss = loss_object(labels, predictions)

        valid_loss(v_loss)
        valid_accuracy(labels, predictions)

    # start training
    for epoch in range(args.epoch):
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()
        step = 0
        for images, labels in train_dataset:
            step += 1
            train_step(images, labels)
            print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch + 1,
                                                                                     args.epoch,
                                                                                     step,
                                                                                     math.ceil(train_count / args.batch_size),
                                                                                     train_loss.result(),
                                                                                     train_accuracy.result()))

        for valid_images, valid_labels in valid_dataset:
            valid_step(valid_images, valid_labels)

        print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
              "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
                                                                  args.epoch,
                                                                  train_loss.result(),
                                                                  train_accuracy.result(),
                                                                  valid_loss.result(),
                                                                  valid_accuracy.result()))

        if (epoch % 1000) == 999:
            savemodel_filepath = os.path.join(args.savemodel_dir, f'{str(epoch + 1)}')
            os.makedirs(savemodel_filepath, exist_ok=True)
            savemodel_file = os.path.join(savemodel_filepath, 'model')
            model.save_weights(filepath=savemodel_file, save_format='tf')
            print(savemodel_filepath)

            # Elapsed time
            end_t = time.time()
            print(f'[Elapsed Time till epoch({str(epoch + 1)}): {str(end_t - start_t)}]', flush=True)


    savemodel_filepath = os.path.join(args.savemodel_dir, 'final')
    os.makedirs(savemodel_filepath, exist_ok=True)
    savemodel_file = os.path.join(savemodel_filepath, 'model')
    model.save_weights(filepath=savemodel_file, save_format='tf')
    print(savemodel_filepath)

    # end time
    end_t = time.time()
    print(f'[Elapsed Time in Total: {str(end_t - start_t)}]', flush=True)
    print(datetime.datetime.now())
