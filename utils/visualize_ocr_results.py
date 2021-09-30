import  os
import argparse
import json 
import cv2 
import numpy as np


def visualize_word_level(ocr_results, imgs_path, output):
    for file_name in os.listdir(ocr_results):
        # open image
        img_path = os.path.join(imgs_path, file_name.replace(".json", '.jpeg') )
        img = cv2.imread(img_path)
        height, width, channels = img.shape
        # print("height ", height)
        # print('width: ', width)
        # open ocr result respectively
        path = os.path.join(ocr_results, file_name)
        print("Procesing: \n\timgs: {} \n\tjson: {}".format(img_path, path))
        fi = open(path, )
        data = json.load(fi)
        word_data = data['WORD']
        # print('word data: ', len(word_data))
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        isClosed = True
        for word in word_data:
            # print('word', word)
            text = word['Text']
            polygon = word['Geometry']['Polygon']
            bbox = word['Geometry']["BoundingBox"]
            left = int(bbox["Left"]*width)
            top= int(bbox["Top"]*height)
            # print('text: ', text)
            img = cv2.putText(img, text, (left, top-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (36,255,12), 1)
            # print('polygon: ', polygon)
            center_coordinates = (left,top)
            img = cv2.circle(img, center_coordinates, radius=2, color=(0, 0, 255), thickness=2)
            img = cv2.circle(img, (int(polygon[2]["X"]*width), int(polygon[2]["Y"]*height)), radius=2, color=(140, 140, 0), thickness=2)

            new_polygon = [[int(polygon[i]["X"]*width), int(polygon[i]["Y"]*height)] for i in range(len(polygon))]
            # print("new polygon ", new_polygon)
            pts = np.array(new_polygon)
            pts = pts.reshape((-1, 1, 2))
            img = cv2.polylines(img, [pts], 
                      isClosed, color, thickness)

        save_path = os.path.join(output, file_name.replace(".json", '.jpeg'))
        cv2.imwrite(save_path, img)
        print("Saved visualize ", file_name)
        fi.close()
        break

def visualize_line_level(ocr_results, imgs_path, output):
    pass

def main(args):
    if args.level == "WORD":
        print("Visualize with word level")
        visualize_word_level(args.ocr_results, args.imgs, args.output)
    elif args.level == "LINE":
        print("Visualize with line level")
        visualize_word_level(args.ocr_results, args.imgs, args.output)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--level", 
        default="WORD", 
        type=str, 
        help="Level: WORD or LINE"
    )
    parser.add_argument(
        "--ocr_results",
        default="/mlcv/Databases/DocVQA_2020-21/task_3/val/infographicVQA_val_v1.0.json",
        type=str,
        help="The path obtain ocr results",
    )
    parser.add_argument(
        "--imgs",
        default="/mlcv/Databases/DocVQA_2020-21/task_3/val/infographicVQA_val_v1.0_images",
        type=str,
        help="The path obtain ocr results",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="The folder save visualize results",
        default="visualizes/ocr_results/val",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_parser()
    main(args)

'''
python utils/visualize_ocr_results.py \
    --level="WORD" \
    --ocr_results="/mlcv/Databases/DocVQA_2020-21/task_3/val/ocr_results" \
    --imgs="/mlcv/Databases/DocVQA_2020-21/task_3/val/infographicVQA_val_v1.0_images" \
    --output="/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/visualizes/ocr_results/val"
'''