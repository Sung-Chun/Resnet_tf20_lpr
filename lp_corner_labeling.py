import os
import json
import cv2
import numpy as np

import glob

# 모든 image 파일에 대해 iteration을 돌렴서
# 1. 해당 이미지의 width, height를 획득하기 (using opencv)
# 2. 해당 json에서 LP의 꼭지점 위치 얻어오기
# 3. LP의 꼭지점 위치에서 사각형을 뽑을건데, 그 사각형이 해당 이미지의 width / height를 벗어나지 않게 조정
# 4. 해당 이미지의 YOLO용 라벨 파일을 생성

LABEL_DIR_PREFIX='d:\\WORK\\LPR\\Resnet_tf20_lpr\\label\\json'
IMAGE_DIR_PREFIX='d:\\WORK\\LPR\\Resnet_tf20_lpr\\original_dataset'
CORNER_IMG_WIDTH=0.1   # 10%
CORNER_IMG_HEIGHT=0.1  # 10%

def imread_hangul_filename(filename):
    np_arr = np.fromfile(filename, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

def do_json(class_text, lp_text):

    json_file = os.path.join(LABEL_DIR_PREFIX, class_text, lp_text + '.json')
    with open(json_file, 'r', encoding="UTF-8") as json_f:
        j_obj = json.load(json_f)
        pass
    lp_corners = j_obj["region"]
    return lp_corners

def do_img(class_text, lp_text, jpg_ext):

    img_file = os.path.join(IMAGE_DIR_PREFIX, class_text, lp_text + jpg_ext)
#    img_obj = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    img_obj = imread_hangul_filename(img_file)
    print(f'[{lp_text}] size:{img_obj.shape},  W({img_obj.shape[1]}) x H({img_obj.shape[0]})')
    return img_obj.shape[1], img_obj.shape[0]

def vertex_box_label(vertex, img_width, img_height):

    center = (vertex[0] / img_width, vertex[1] / img_height)

    x1 = center[0] - (CORNER_IMG_WIDTH / 2.)
    if x1 < 0.0: x1 = 0.0
    x2 = x1 + CORNER_IMG_WIDTH
    if x2 >= 1.0: x2 = 0.99

    y1 = center[1] - (CORNER_IMG_HEIGHT / 2.)
    if y1 < 0.0: y1 = 0.0
    y2 = y1 + CORNER_IMG_HEIGHT
    if y2 >= 1.0: y2 = 0.99

    '''
    x1 = vertex[0] - (CORNER_IMG_WIDTH / 2.)
    if x1 < 0: x1 = 0
    x2 = x1 + CORNER_IMG_WIDTH
    if x2 >= img_width: x2 = img_width - 1
    y1 = vertex[1] - (CORNER_IMG_HEIGHT / 2)
    if y1 < 0: y1 = 0
    y2 = y1 + CORNER_IMG_HEIGHT
    if y2 >= img_height: y2 = img_height - 1
    '''

    vertex_class = ''
    size = (x2 - x1, y2 - y1)

    return {'class': vertex_class, 'center': center, 'size': size}

def do_something(class_text, lp_text, jpg_ext, label_txt_output_dir):
    lp_corners = do_json(class_text, lp_text)
    img_width, img_height = do_img(class_text, lp_text, jpg_ext)

    label_txt_output_file = os.path.join(label_txt_output_dir, lp_text + '.txt')
    with open(label_txt_output_file, 'w') as f_label_txt:
        # LeftUp vertex
        label = vertex_box_label(lp_corners['LeftUp'], img_width, img_height)
        f_label_txt.write(f'0 {label["center"][0]:.6f} {label["center"][1]:.6f} {label["size"][0]:.6f} {label["size"][1]:.6f}\n')
        # RightUp vertex
        label = vertex_box_label(lp_corners['RightUp'], img_width, img_height)
        f_label_txt.write(f'0 {label["center"][0]:.6f} {label["center"][1]:.6f} {label["size"][0]:.6f} {label["size"][1]:.6f}\n')
        # RightDown vertex
        label = vertex_box_label(lp_corners['RightDown'], img_width, img_height)
        f_label_txt.write(f'0 {label["center"][0]:.6f} {label["center"][1]:.6f} {label["size"][0]:.6f} {label["size"][1]:.6f}\n')
        # LeftDown vertex
        label = vertex_box_label(lp_corners['LeftDown'], img_width, img_height)
        f_label_txt.write(f'0 {label["center"][0]:.6f} {label["center"][1]:.6f} {label["size"][0]:.6f} {label["size"][1]:.6f}\n')

if __name__ == '__main__':

    walk_dir = './original_dataset'
    label_txt_output_dir = './label/txt'
    os.makedirs(label_txt_output_dir, exist_ok=True)

    i = 0
    jpg_file_list = glob.iglob(walk_dir + '/*/*.jpg*', recursive=True)
    for jpg_file_path in jpg_file_list:
        (folder_path, jpg_file_name) = os.path.split(jpg_file_path)
        (_, class_folder) = os.path.split(folder_path)
        (lp_text, file_ext) = os.path.splitext(jpg_file_name)
        if (file_ext != '.jpg') and (file_ext != '.JPG'):
            continue

        do_something(class_folder, lp_text, file_ext, label_txt_output_dir)

        i += 1
