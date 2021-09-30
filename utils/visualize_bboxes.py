import cv2 
import os
import argparse
import tqdm
import numpy as np

FILE_NAME="10022"



def visualize_info_feature(imgs_folder, info_features_folder, visualize_output):
    
    for img_name in os.listdir(imgs_folder):
        img_path = os.path.join(imgs_folder, img_name)
        print("img path: ", img_path)
        info_name = img_name.replace('.jpeg', "_info.npy")
        info_path = os.path.join(info_features_folder, info_name)
        print("info path: ", info_path)
        #open image 
        img = cv2.imread(img_path)
        # print('img: ', img)
        #open info file 
        info = np.load(info_path, allow_pickle=True)
        info = info.item()
        bboxes = info["bbox"]
        # print('bboxes: ', bboxes)
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        for i in range(100):
            bbox = bboxes[i]
            start_point = (int(bbox[0]), int(bbox[1]))
            end_point = (int(bbox[2]), int(bbox[3]))
            # print("bbox: ", bbox)
            # print('start point ', start_point )
            # print("end point ", end_point)
            img = cv2.rectangle(img, start_point, end_point, color, thickness)

        save_path = os.path.join(visualize_output, img_name)
        cv2.imwrite(save_path, img)
        print("Saved objs at ", save_path)


def main(parse):
    # info_features_folder = "extracted_output"
    # imgs_folder = "/mlcv/Databases/DocVQA_2020-21/task_3/val/infographicVQA_val_v1.0_images"
    # visualize_output = "visualizes/val"
    print(parse.info_features)
    print(parse.imgs_folder)
    print(parse.output)
    visualize_info_feature(parse.imgs_folder, parse.info_features, parse.output)

def get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--info_features", default="extracted_output", type=str, help="Folder save info of object features"
        )
        parser.add_argument(
            "--imgs_folder",
            default="/mlcv/Databases/DocVQA_2020-21/task_3/val/infographicVQA_val_v1.0_images",
            type=str,
            help="Images folder where save image",
        )
        parser.add_argument(
            "--output",
            type=str,
            help="folder save visualize results",
            default="visualizes",
        )
        return parser.parse_args()

if __name__ == "__main__":
    parse = get_parser()
    main(parse)

'''
python utils/visualize_bboxes.py \
    --info_features="extracted_output" \
    --imgs_folder="/mlcv/Databases/DocVQA_2020-21/task_3/val/infographicVQA_val_v1.0_images" \
    --output="visualizes/val"
'''

