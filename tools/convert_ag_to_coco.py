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
    annotations_train = {"info": [], "licenses": [], "images": [], "annotations": [], "categories": []}
    annotations_val = {"info": [], "licenses": [], "images": [], "annotations": [], "categories": []}

    object_classes = {}
    with open(op.join(args.root_dir, "annotations/object_classes.txt"), "r") as file:
        lines = file.readlines()
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

    frame_list = []
    with open(op.join(args.root_dir, "annotations/frame_list.txt"), "r") as file:
        lines = file.readlines()
        frame_list = [line.rstrip() for line in lines]

    object_bbox_and_relationship = pickle.load(
        open(
            op.join(args.root_dir, "annotations/object_bbox_and_relationship.pkl"),
            "rb",
        )
    )
    person_bbox = pickle.load(open(op.join(args.root_dir, "annotations/person_bbox.pkl"), "rb"))

    if not op.exists(op.join(args.dst_dir, "train")):
        os.mkdir(op.join(args.dst_dir, "train"))
    if not op.exists(op.join(args.dst_dir, "val")):
        os.mkdir(op.join(args.dst_dir, "val"))
    if not op.exists(op.join(args.dst_dir, "annotations")):
        os.mkdir(op.join(args.dst_dir, "annotations"))

    count_id = 0
    image_id = 0
    print("Start process")
    for frame_id in tqdm(frame_list):
        video_id, frame_name = frame_id.split("/")
        frame_path = f"{video_id}/{op.splitext(frame_name)[0]}.jpg"
        person = person_bbox[frame_id]
        rels = object_bbox_and_relationship[frame_id]

        image_set = ""
        for rel in rels:
            image_set = rel["metadata"]["set"]

            box = rel["bbox"] # (xywh)
            if box is None:
                continue
            annot = {
                "segmentation": [],
                "area": 0,
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": list(np.array(box, dtype=np.float64)),
                "category_id": object_classes[rel["class"].replace("/", "")],
                "id": count_id,
            }
            if image_set == "train":
                annotations_train["annotations"].append(annot)
            else:
                annotations_val["annotations"].append(annot)
            count_id += 1

        if image_set == "":
            continue

        if person["bbox"].shape[0] == 0:
            continue

        box = person["bbox"][0] # (xyxy)

        annot = {
            "segmentation": [],
            "area": 0,
            "iscrowd": 0,
            "image_id": image_id,
            "bbox": list_box_xyxy_to_xywh(list(np.array(box, dtype=np.float64))),
            "category_id": object_classes["person"],
            "id": count_id,
        }
        if image_set == "train":
            annotations_train["annotations"].append(annot)
        else:
            annotations_val["annotations"].append(annot)
        count_id += 1

        # copy frame
        image_info = {
            "license": 1,
            "file_name": f"{image_id}.jpg",
            "coco_url": "",
            "height": person["bbox_size"][1],
            "width": person["bbox_size"][0],
            "date_captured": "",
            "flickr_url": "",
            "id": image_id,
        }
        if image_set == "train":
            annotations_train["images"].append(image_info)
        else:
            annotations_val["images"].append(image_info)
        # copy frame
        if args.copy_frame:
            frame_path = op.join(args.root_dir, "all_frames", frame_path)
            dst_path = op.join(args.dst_dir, image_set if image_set == "train" else "val", f"{image_id}.jpg")
            if args.override or not op.exists(dst_path):
                shutil.copy(frame_path, dst_path)
        image_id += 1

    json.dump(annotations_train, open(os.path.join(args.dst_dir, "annotations/instances_train2017.json"), "w"))
    json.dump(annotations_val, open(os.path.join(args.dst_dir, "annotations/instances_val2017.json"), "w"))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
