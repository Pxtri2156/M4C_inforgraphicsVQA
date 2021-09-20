import cv2 
import os
import tqdm


FILE_NAME="10022"



def visualize_info_feature(imgs_folder, info_features_folder):
    
    for img_name in os.listdir(imgs_folder):
        img_path = os.path.join(imgs_folder, img_name)
        print("img path: ", img_path)
        info_name = img_name.replace('.jpeg', "_info.npy")
        info_path = os.path.join(info_features_folder, info_name)
        print("info path: ", info_path)
        break


def main():
    info_features_folder = "/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/extracted_output"
    imgs_folder = "/mlcv/Databases/DocVQA_2020-21/task_3/val/infographicVQA_val_v1.0_images"
    visualize_output = ""
    visualize_info_feature(imgs_folder, info_features_folder)
if __name__ == "__main__":
    main()