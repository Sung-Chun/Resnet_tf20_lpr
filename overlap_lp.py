import tensorflow as tf
import config
from prepare_data import generate_test_datasets, generate_test_corner_datasets, imread_hangul_filename, imwrite_hangul_filename
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
    parser.add_argument('--savemodel-dir', type=str, required=True, help='directory to save model')
    parser.add_argument('--overlap-lp-dir', type=str, required=True, help='directory of LP to overlap')
    parser.add_argument('--saveimage-dir', type=str, required=True, help='directory to save overlapped image')

    return parser

def compute_iou(corners1, corners2):

    x_min = np.min([corners1[:,0], corners2[:,0]])
    x_max = np.max([corners1[:,0], corners2[:,0]])
    y_min = np.min([corners1[:,1], corners2[:,1]])
    y_max = np.max([corners1[:,1], corners2[:,1]])

    aou, aoo = 0, 0
    for y in range(y_min, y_max+1):
        for x in range(x_min, x_max+1):
            r1 = (cv2.pointPolygonTest(corners1, (x, y), False) > 0.0)
            r2 = (cv2.pointPolygonTest(corners2, (x, y), False) > 0.0)
            r = r1 + r2
            if r >= 1:
                aou += 1
                if r == 2:  aoo += 1
    return aoo / aou

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

    # load the overlap image
    if os.path.exists(args.overlap_lp_dir) is False:
        print(f' ERROR: the overlap lp directory does not exist. ({args.overlap_lp_dir})')
        os.exit()
    dir_list = os.listdir(args.overlap_lp_dir)
    lp_label_dir_list = [x for x in dir_list if os.path.isdir(os.path.join(args.overlap_lp_dir, x))]
    overlap_lp_img_dict = dict()
    for lp_label in lp_label_dir_list:
        dir_list = os.listdir(os.path.join(args.overlap_lp_dir, lp_label))
        overlap_img_list = [x for x in dir_list if not os.path.isdir(x)]
        if len(overlap_img_list) == 0:
            continue
        overlap_lp_img_dict[lp_label] = imread_hangul_filename(os.path.join(args.overlap_lp_dir, lp_label, overlap_img_list[0]))

    # load the model
    model = get_model(args.model, args.width, args.height, channels=3, regressor=True)
    savemodel_filepath = os.path.join(args.savemodel_dir, args.epoch, 'model')
    model.load_weights(filepath=savemodel_filepath)

    # get the original_dataset
    test_dataset, test_count, all_image_path, all_image_label = generate_test_corner_datasets(8, args.width, args.height)
    i = 0
    ok_count = fail_count = 0
    for test_image, test_label in test_dataset:
        img_obj = imread_hangul_filename(all_image_path[i])
        wd, hi = img_obj.shape[1], img_obj.shape[0]

        # Box coordinates of predictions
        predictions = model(test_image, training=False).numpy()[0]
        pred_corners = np.array([[round(predictions[0] * wd), round(predictions[1] * hi)],
                  [round(predictions[2] * wd), round(predictions[3] * hi)],
                  [round(predictions[4] * wd), round(predictions[5] * hi)],
                  [round(predictions[6] * wd), round(predictions[7] * hi)]], np.int32)

        # Box coordinates of true labels
        true_pts = test_label.numpy()[0]
        true_corners = np.array([[round(true_pts[0] * wd), round(true_pts[1] * hi)],
                  [round(true_pts[2] * wd), round(true_pts[3] * hi)],
                  [round(true_pts[4] * wd), round(true_pts[5] * hi)],
                  [round(true_pts[6] * wd), round(true_pts[7] * hi)]], np.int32)

        cv2.polylines(img_obj, [true_corners], True, (100, 50, 250), 2)
        cv2.polylines(img_obj, [pred_corners], True, (100, 255, 100), 2)

        # IOU를 계산한다.
        iou = compute_iou(true_corners, pred_corners)

        # 이미지 경로에서 LP Label을 얻어낸다.
        lp_label, lp_img_filename = all_image_path[i].split(os.sep)[2:4]

        # pts_src: 합성할 번호판의 꼭지점 좌표
        # pts_dst: 대상 이미지의 번호판 꼭지점 좌표 (pred_corners)
        overlap_lp_img = overlap_lp_img_dict[lp_label]
        overlap_hi, overlap_wd, _ = overlap_lp_img.shape
        pts_src = np.float32([[0, 0], [overlap_wd, 0], [0, overlap_hi], [overlap_wd, overlap_hi]])
        pts_dst = np.float32([pred_corners[0], pred_corners[1], pred_corners[3], pred_corners[2]])

        # Transform matrix를 구하고, Transform을 실행한다.
        M_xform = cv2.getPerspectiveTransform(pts_src, pts_dst)
        overlap_lp_warped = cv2.warpPerspective(overlap_lp_img, M_xform, (wd, hi))

        # Transform 된 이미지에서 까만 부분에 대해 마스크를 만들어낸다.
        overlap_lp_warped_gray = cv2.cvtColor(overlap_lp_warped, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(overlap_lp_warped_gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # 대상 이미지 (background 이미지) 에서 덮어 쓸 부분을 까맣게 만든다.
        img1_bg = cv2.bitwise_and(img_obj, img_obj, mask=mask_inv)

        # Transform 된 이미지 (foreground 이미지) 에서 덮어 쓸 부분을 까맣게 만든다.
        img2_fg = cv2.bitwise_and(overlap_lp_warped, overlap_lp_warped, mask=mask)

        # 두 개 이미지를 합친다.
        img_obj = cv2.add(img1_bg, img2_fg)

        # 합친 이미지를 파일로 저장
        os.makedirs(os.path.join(args.saveimage_dir, lp_label), exist_ok=True)
        save_filename = os.path.join(args.saveimage_dir, lp_label, lp_img_filename)
        if imwrite_hangul_filename(save_filename, img_obj):
            print(f'  Completed saving image: [ \033[0;32;40m{save_filename}\033[0m ], iou={iou:.2f}')
        else:
            print(f'  Failed in saving image: [ \033[0;31;40m{save_filename}\033[0m ], iou={iou:.2f}')

        i += 1


    print(f'----------------------------------')
    print(f'  Total test samples: {test_count}')
    print(f'     OK count  : {ok_count}')
    print(f'     FAIL count: {fail_count}')
    print(f'  Test accuracy: \033[0;32;40m{(100 * ok_count / test_count):.2f}%\033[0m')
