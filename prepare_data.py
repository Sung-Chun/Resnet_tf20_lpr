import tensorflow as tf
import config
import pathlib
from config import image_height, image_width, channels

import os
import json

def load_and_preprocess_image(img_path, img_width, img_height, num_channels):
    # read pictures
    img_raw = tf.io.read_file(img_path)
    # decode pictures
    img_tensor = tf.image.decode_jpeg(img_raw, channels=num_channels)
    # resize
    img_tensor = tf.image.resize(img_tensor, [img_height, img_width])
    img_tensor = tf.cast(img_tensor, tf.float32)
    # normalization
    img = img_tensor / 255.0
    return img

def get_images_and_labels(data_root_dir):
    # get all images' paths (format: string)
    data_root = pathlib.Path(data_root_dir)
    all_image_path = [str(path) for path in list(data_root.glob('*/*'))]
    # get labels' names
    label_names = sorted(item.name for item in data_root.glob('*/'))
    # dict: {label : index}
    label_to_index = dict((label, index) for index, label in enumerate(label_names))
    print('---------------------------------')
    print('  Label-to-index:')
    print(label_to_index)
    print('---------------------------------')
    # get all images' labels
    all_image_label = [label_to_index[pathlib.Path(single_image_path).parent.name] for single_image_path in all_image_path]

    return all_image_path, all_image_label


def get_dataset(dataset_root_dir, img_width, img_height):
    all_image_path, all_image_label = get_images_and_labels(data_root_dir=dataset_root_dir)
    # print("image_path: {}".format(all_image_path[:]))
    # print("image_label: {}".format(all_image_label[:]))
    # load the dataset and preprocess images
    image_dataset = tf.data.Dataset.from_tensor_slices(all_image_path).map(lambda x: load_and_preprocess_image(x, img_width, img_height, 1))
    label_dataset = tf.data.Dataset.from_tensor_slices(all_image_label)
    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
    image_count = len(all_image_path)

    return dataset, image_count, all_image_path, all_image_label


def generate_datasets(batch_size, img_width, img_height):
    train_dataset, train_count, _, _ = get_dataset(config.train_dir, img_width, img_height)
    valid_dataset, valid_count, _, _ = get_dataset(config.valid_dir, img_width, img_height)
    test_dataset, test_count, _, _ = get_dataset(config.test_dir, img_width, img_height)


    # read the original_dataset in the form of batch
    train_dataset = train_dataset.shuffle(buffer_size=train_count).batch(batch_size)
    valid_dataset = valid_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count

def generate_test_datasets(batch_size, img_width, img_height):
    test_dataset, test_count, all_image_path, all_image_label= get_dataset(config.test_dir, img_width, img_height)

    # read the original_dataset in the form of batch
    test_dataset = test_dataset.batch(1)

    return test_dataset, test_count, all_image_path, all_image_label


###########
## functions for corner coordinate data
import cv2
import numpy as np
def imread_hangul_filename(filename):
    np_arr = np.fromfile(filename, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

def imwrite_hangul_filename(filename, img_obj):
    try:
        ext = os.path.splitext(filename)[-1]
        result, n = cv2.imencode(ext, img_obj)
        if not result:
            return False
        with open(filename, mode='w+b') as f:
            n.tofile(f)
        return True
    except Exception as e:
        print(e)
        return False

def corner_coord(img_path):
    img_obj = imread_hangul_filename(img_path)

    folder_splits = img_path.split(os.sep)
    (lp_text, file_ext) = os.path.splitext(folder_splits[-1])

    # json file path is assumed to be in 'label' folder of current path
    json_file_path = os.path.join('label', 'json', folder_splits[-2], lp_text + '.json')
    with open(json_file_path, 'r', encoding="UTF-8") as json_f:
        j_obj = json.load(json_f)
        pass
    lp_corners = j_obj["region"]
    return [lp_corners['LeftUp'][0] / img_obj.shape[1], lp_corners['LeftUp'][1] / img_obj.shape[0],
            lp_corners['RightUp'][0] / img_obj.shape[1], lp_corners['RightUp'][1] / img_obj.shape[0],
            lp_corners['RightDown'][0]/ img_obj.shape[1], lp_corners['RightDown'][1] / img_obj.shape[0],
            lp_corners['LeftDown'][0] / img_obj.shape[1], lp_corners['LeftDown'][1] / img_obj.shape[0]]

def get_images_and_corner_labels(data_root_dir):
    # get all images' paths (format: string)
    data_root = pathlib.Path(data_root_dir)
    all_image_path = [str(path) for path in list(data_root.glob('*/**/*.jpg'))]

    # get corner coordinate of all images
    all_image_label = [corner_coord(single_image_path) for single_image_path in all_image_path]

    return all_image_path, all_image_label

def get_corner_dataset(dataset_root_dir, img_width, img_height):
    all_image_path, all_image_label = get_images_and_corner_labels(data_root_dir=dataset_root_dir)
    # print("image_path: {}".format(all_image_path[:]))
    # print("image_label: {}".format(all_image_label[:]))
    # load the dataset and preprocess images
    image_dataset = tf.data.Dataset.from_tensor_slices(all_image_path).map(load_and_preprocess_image)
    label_dataset = tf.data.Dataset.from_tensor_slices(all_image_label)
    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
    image_count = len(all_image_path)

    return dataset, image_count, all_image_path, all_image_label

def generate_corner_datasets(batch_size, img_width, img_height):
    train_dataset, train_count, _, _ = get_corner_dataset(config.train_dir, img_width, img_height)
    valid_dataset, valid_count, _, _ = get_corner_dataset(config.valid_dir, img_width, img_height)
    test_dataset, test_count, _, _ = get_corner_dataset(config.test_dir, img_width, img_height)


    # read the original_dataset in the form of batch
    train_dataset = train_dataset.shuffle(buffer_size=train_count).batch(batch_size)
    valid_dataset = valid_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count

def generate_test_corner_datasets(batch_size, img_width, img_height):
    test_dataset, test_count, all_image_path, all_image_label= get_corner_dataset(config.test_dir, img_width, img_height)

    # read the original_dataset in the form of batch
    test_dataset = test_dataset.batch(1)

    return test_dataset, test_count, all_image_path, all_image_label
