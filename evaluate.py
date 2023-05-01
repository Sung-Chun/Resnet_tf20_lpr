import tensorflow as tf
import config
from prepare_data import generate_datasets
from train import get_model

import os, sys
import argparse

def get_argparser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-B', '--batch-size', type=int, default=8)
    parser.add_argument('--width', type=int, default=224, help='image width')
    parser.add_argument('--height', type=int, default=128, help='image height')
    parser.add_argument('--model', type=str, default='resnet50', help='resnet model name')
    parser.add_argument('--savemodel-dir', type=str, required=True, help='directory to save model')

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

    # get the original_dataset
    train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets(args.batch_size, args.width, args.height)
    # print(train_dataset)
    # load the model
    model = get_model(args.model, args.width, args.height, channels=1)
    savemodel_filepath = os.path.join(args.savemodel_dir, 'final', 'model')
    model.load_weights(filepath=savemodel_filepath)

    # Get the accuracy on the test set
    loss_object = tf.keras.metrics.SparseCategoricalCrossentropy()
    test_loss = tf.keras.metrics.Mean()
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)

        # 여기에 labels와 실제 predictions의 결과를 화면에 보여준다...

        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    for test_images, test_labels in test_dataset:
        test_step(test_images, test_labels)
        print("loss: {:.5f}, test accuracy: {:.5f}".format(test_loss.result(),
                                                           test_accuracy.result()))

    print("The accuracy on test set is: {:.3f}%".format(test_accuracy.result()*100))