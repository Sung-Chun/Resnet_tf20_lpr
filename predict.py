import tensorflow as tf
import config
from prepare_data import generate_test_datasets, generate_test_corner_datasets, imread_hangul_filename
from train import get_model

import os, sys
import ctypes
import argparse
import numpy as np

import cv2

def enable_vt100():
    kernel32 = ctypes.WinDLL('kernel32')
    hStdOut = kernel32.GetStdHandle(-11)
    mode = ctypes.c_ulong()
    kernel32.GetConsoleMode(hStdOut, ctypes.byref(mode))
    mode.value |= 4
    kernel32.SetConsoleMode(hStdOut, mode)

def get_argparser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--width', type=int, default=224, help='image width')
    parser.add_argument('--height', type=int, default=128, help='image height')
    parser.add_argument('--epoch', type=str, default='final', help='model to use')
    parser.add_argument('--model', type=str, default='resnet50', help='resnet model name')
    parser.add_argument('--regression', type=bool, default=False, help='resnet classifier/regressor')
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

    if os.sep == '\\':
        enable_vt100()

    # load the model
    model = get_model(args.model, args.width, args.height, channels=3, regressor=args.regression)
    savemodel_filepath = os.path.join(args.savemodel_dir, args.epoch, 'model')
    model.load_weights(filepath=savemodel_filepath)

    # get the original_dataset
    if args.regression:
        test_dataset, test_count, all_image_path, all_image_label = generate_test_corner_datasets(8, args.width, args.height)
        i = 0
        ok_count = fail_count = 0
        for test_image, test_label in test_dataset:
            img_obj = imread_hangul_filename(all_image_path[i])
            w, h = img_obj.shape[1], img_obj.shape[0]

            # Box coordinates of predictions
            predictions = model(test_image, training=False).numpy()[0]
            pred_corners = np.array([[round(predictions[0] * w), round(predictions[1] * h)],
                      [round(predictions[2] * w), round(predictions[3] * h)],
                      [round(predictions[4] * w), round(predictions[5] * h)],
                      [round(predictions[6] * w), round(predictions[7] * h)]], np.int32)

            # Box coordinates of true labels
            true_pts = test_label.numpy()[0]
            true_corners = np.array([[round(true_pts[0] * w), round(true_pts[1] * h)],
                      [round(true_pts[2] * w), round(true_pts[3] * h)],
                      [round(true_pts[4] * w), round(true_pts[5] * h)],
                      [round(true_pts[6] * w), round(true_pts[7] * h)]], np.int32)

            cv2.polylines(img_obj, [true_corners], True, (100, 50, 250), 2)
            cv2.polylines(img_obj, [pred_corners], True, (100, 255, 100), 2)


            cv2.imshow("plate_box", img_obj)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            i += 1

    else:
        test_dataset, test_count, all_image_path, all_image_label = generate_test_datasets(8, args.width, args.height)
        i = 0
        ok_count = fail_count = 0
        for test_image, test_label in test_dataset:
            predictions = model(test_image, training=False)
            pred_label = np.argmax(predictions.numpy())
            true_label = test_label[0].numpy()
            if pred_label == true_label:
                print(f'-({i}) {all_image_path[i]}      \033[0;32;40m[OK]\033[0m, True: {true_label}, Pred: {pred_label}')
                ok_count += 1
            else:
                print(f'-({i}) {all_image_path[i]}      \033[0;31;40m[FAIL]\033[0m, True: {true_label}, Pred: {pred_label}')
                fail_count += 1
            i += 1

    print(f'----------------------------------')
    print(f'  Total test samples: {test_count}')
    print(f'     OK count  : {ok_count}')
    print(f'     FAIL count: {fail_count}')
    print(f'  Test accuracy: \033[0;32;40m{(100 * ok_count / test_count):.2f}%\033[0m')
