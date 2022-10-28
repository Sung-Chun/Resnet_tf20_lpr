import os
import json
import cv2


# 모든 image 파일에 대해 iteration을 돌렴서
# 1. 해당 이미지의 width, height를 획득하기 (using opencv)
# 2. 해당 json에서 LP의 꼭지점 위치 얻어오기
# 3. LP의 꼭지점 위치에서 사각형을 뽑을건데, 그 사각형이 해당 이미지의 width / height를 벗어나지 않게 조정
# 4. 해당 이미지의 YOLO용 라벨 파일을 생성

DIR_PREFIX='/mnt/LPR/dataset'
CORNER_IMG_WIDTH=8
CORNER_IMG_HEIGHT=8

def do_json():

    json_file = os.path.join(DIR_PREFIX, 'json', '1996_n', '경기32도9430.json')
    with open(json_file, 'r', encoding="UTF-8") as json_f:
        j_obj = json.load(json_f)
        pass
    lp_corners = j_obj["region"]
    return lp_corners

def do_img():

    img_file = os.path.join(DIR_PREFIX, 'image', '1996_n', '경기32도9430.jpg')
    img_obj = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    print(f'Size info: {img_obj.shape},  W({img_obj.shape[1]}) x H({img_obj.shape[0]})')
    return img_obj.shape[1], img_obj.shape[0]

def vertex_box_label(vertex, img_width, img_height):
    x1 = vertex[0] - (CORNER_IMG_WIDTH / 2)
    if x1 < 0: x1 = 0
    x2 = x1 + CORNER_IMG_WIDTH
    if x2 >= img_width: x2 = img_width - 1
    y1 = vertex[1] - (CORNER_IMG_HEIGHT / 2)
    if y1 < 0: y1 = 0
    y2 = y1 + CORNER_IMG_HEIGHT
    if y2 >= img_height: y2 = img_height - 1

    vertex_class = '0'
    center = (vertex[0] / img_width, vertex[1] / img_height)
    size = ((x2 - x1) / img_width, (y2 - y1) / img_height)
#    return [vertex_class, center, size]
    return {'class': vertex_class, 'center': center, 'size': size}

def do_something():
    lp_corners = do_json()
    img_width, img_height = do_img()

    label = dict()

    # LeftUp vertex
    label = vertex_box_label(lp_corners['LeftUp'], img_width, img_height)
    print(f'{label["center"]}, {label["size"]}')
    # RightUp vertex
    label = vertex_box_label(lp_corners['RightUp'], img_width, img_height)
    print(f'{label["center"]}, {label["size"]}')
    # RightDown vertex
    label = vertex_box_label(lp_corners['RightDown'], img_width, img_height)
    print(f'{label["center"]}, {label["size"]}')
    # LeftDown vertex
    label = vertex_box_label(lp_corners['LeftDown'], img_width, img_height)
    print(f'{label["center"]}, {label["size"]}')

if __name__ == '__main__':
    do_something()

