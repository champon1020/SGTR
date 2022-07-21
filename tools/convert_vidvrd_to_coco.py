import argparse
import json
import os
import os.path as op
import pickle
import shutil

import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", required=True, type=str)
parser.add_argument("--dst_dir", required=True, type=str)
parser.add_argument("--copy_frame", action="store_true")
parser.add_argument("--override", action="store_true")


def list_box_xyxy_to_xywh(x):
    x0, y0, x1, y1 = x
    b = [x0, y0, (x1 - x0), (y1 - y0)]
    return b


def main(args):
    print("Start process")
    annotations_train = {"info": [], "licenses": [], "images": [], "annotations": [], "categories": []}
    annotations_val = {"info": [], "licenses": [], "images": [], "annotations": [], "categories": []}

    object_classes = {}
    with open(op.join(args.root_dir, "annotations/object.txt"), "r") as fp:
        lines = fp.readlines()
        class_id = 0
        for line in lines:
            class_name = line.rstrip()
            object_classes[class_name] = class_id
            categories = {
                "supercategory": class_name,
                "id": class_id,
                "name": class_name,
            }
            annotations_train["categories"].append(categories)
            annotations_val["categories"].append(categories)
            class_id += 1

    if not op.exists(op.join(args.dst_dir, "annotations")):
        os.mkdir(op.join(args.dst_dir, "annotations"))

    annotation_dirname_train = op.join(args.root_dir, "train")
    annotation_files_train = os.listdir(annotation_dirname_train)
    annotation_dirname_val = op.join(args.root_dir, "test")
    annotation_files_val = os.listdir(annotation_dirname_val)

    def format_annotation(image_set, annotation_files, annotation_dirname, start_count_id = 0, start_image_id=0):
        print(f"Start process {image_set}")
        if not op.exists(op.join(args.dst_dir, image_set)):
            os.mkdir(op.join(args.dst_dir, image_set))
        count_id = start_count_id
        image_id = start_image_id
        for file_name in tqdm(annotation_files):
            data = json.load(open(op.join(annotation_dirname, file_name)))
            width, height = data["width"], data["height"]
            obj_tid_to_label = {o["tid"]: o["category"] for o in data["subject/objects"]}

            for fid, frame in enumerate(data["trajectories"]):
                if len(frame) == 0:
                    continue
                updated = False
                frame_id = f"{image_id}.jpg"
                for ind, bbox in enumerate(frame):
                    if bbox is None:
                        continue
                    box = [
                        bbox["bbox"]["xmin"],
                        bbox["bbox"]["ymin"],
                        bbox["bbox"]["xmax"],
                        bbox["bbox"]["ymax"],
                    ]
                    annot = {
                        "segmentation": [],
                        "area": 0,
                        "iscrowd": 0,
                        "image_id": image_id,
                        "bbox": list_box_xyxy_to_xywh(box),
                        "category_id": object_classes[obj_tid_to_label[bbox["tid"]]],
                        "id": count_id,
                    }
                    count_id += 1
                    if image_set == "train":
                        annotations_train["annotations"].append(annot)
                    elif image_set == "val":
                        annotations_val["annotations"].append(annot)
                    updated = True

                if not updated:
                    continue
                image_info = {
                    "license": 1,
                    "file_name": frame_id,
                    "coco_url": "",
                    "height": height,
                    "width": width,
                    "date_captured": "",
                    "flickr_url": "",
                    "id": image_id,
                }
                if image_set == "train":
                    annotations_train["images"].append(image_info)
                elif image_set == "val":
                    annotations_val["images"].append(image_info)
                # copy frame
                if args.copy_frame:
                    frame_path = op.join(args.root_dir, "sampled_frames", frame_id)
                    dst_path = op.join(args.dst_dir, image_set, f"{image_id}.jpg")
                    if args.override or not op.exists(dst_path):
                        shutil.copy(frame_path, dst_path)
                image_id += 1
        return count_id, image_id

    count_id, image_id = format_annotation("train", annotation_files_train, annotation_dirname_train, 0, 0)
    count_id, image_id = format_annotation("val", annotation_files_val, annotation_dirname_val, count_id, image_id)

    json.dump(annotations_train, open(op.join(args.dst_dir, "annotations/instances_train2017.json"), "w"))
    json.dump(annotations_val, open(op.join(args.dst_dir, "annotations/instances_val2017.json"), "w"))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
