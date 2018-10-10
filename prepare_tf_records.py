import os
from random import shuffle
import argparse

import tensorflow as tf

from PIL import Image

from object_detection.utils import dataset_util

from utils.annotations import AnnotationParser

import settings


class TfRecordsCreator:

    def __init__(self, image_folder, annotation_folder):
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder

    def create_tf_example(self, image_path, annotation_path):
        img = Image.open(image_path)
        encoded = bytes(tf.gfile.GFile(image_path, "rb").read())
        bounding_boxes = AnnotationParser(annotation_path).parse_annotations()
        return tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image/height': dataset_util.int64_feature(img.height),
                    'image/width': dataset_util.int64_feature(img.width),
                    'image/filename': dataset_util.bytes_feature(str.encode(img.filename)),
                    'image/source_id': dataset_util.bytes_feature(str.encode(img.filename)),
                    'image/encoded': dataset_util.bytes_feature(encoded),
                    'image/format': dataset_util.bytes_feature(str.encode(img.format)),
                    'image/object/bbox/xmin': dataset_util.float_list_feature([b.xmin for b in bounding_boxes]),
                    'image/object/bbox/xmax': dataset_util.float_list_feature([b.xmax for b in bounding_boxes]),
                    'image/object/bbox/ymin': dataset_util.float_list_feature([b.ymin for b in bounding_boxes]),
                    'image/object/bbox/ymax': dataset_util.float_list_feature([b.ymax for b in bounding_boxes]),
                    'image/object/class/text': dataset_util.bytes_list_feature([str.encode(b.class_name) for b in bounding_boxes]),
                    'image/object/class/label': dataset_util.int64_list_feature([b.class_id for b in bounding_boxes]),
                }
            )
        )

    def transform_dataset(self, shuffle_dataset=True):
        image_files = os.listdir(self.image_folder)
        if shuffle_dataset:
            shuffle(image_files)
        split_index = int(len(image_files)*(1-settings.TEST_SIZE))
        with open(settings.CLASSES_FILE, 'w') as f:
            for class_name, class_id in settings.CLASSES.items():
                item = ("item {\n"
                        "\tid: " + str(class_id) + "\n"
                        "\tname: ’" + class_name + "’\n"
                        "}\n")
                f.write(item)
        with tf.python_io.TFRecordWriter(settings.TRAIN_RECORD) as writer:
            for image_filename in image_files[:split_index]:
                tf_example = self.create_tf_example(
                    os.path.join(self.image_folder, image_filename),
                    os.path.join(self.annotation_folder, '%s.xml' % image_filename.split('.')[0]),
                )
                writer.write(tf_example.SerializeToString())
        with tf.python_io.TFRecordWriter(settings.TEST_RECORD) as writer:
            for image_filename in image_files[split_index:]:
                tf_example = self.create_tf_example(
                    os.path.join(self.image_folder, image_filename),
                    os.path.join(self.annotation_folder, '%s.xml' % image_filename.split('.')[0]),
                )
                writer.write(tf_example.SerializeToString())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', type=str)
    parser.add_argument('-a', '--annotation_path', type=str)
    args = parser.parse_args()

    assert args.image_path is not None
    assert args.annotation_path is not None

    tf_records_creator = TfRecordsCreator(args.image_path, args.annotation_path)
    tf_records_creator.transform_dataset(shuffle_dataset=True)