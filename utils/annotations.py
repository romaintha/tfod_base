import xml.etree.ElementTree as ET

from settings import CLASSES


class AnnotationParser:

    def __init__(self, annotation_path):
        self.annotation_path = annotation_path

    def parse_annotations(self):
        tree = ET.parse(self.annotation_path)
        root = tree.getroot()

        bounding_boxes = []
        for object in root.findall('object'):
            xmin = int(object.find('bndbox').find('xmin').text)
            xmax = int(object.find('bndbox').find('xmax').text)
            ymin = int(object.find('bndbox').find('ymin').text)
            ymax = int(object.find('bndbox').find('ymax').text)
            class_name = object.find('name').text

            bounding_boxes.append(
                BoundingBox(xmin, xmax, ymin, ymax, class_name)
            )
        return bounding_boxes


class BoundingBox:

    def __init__(self, xmin, xmax, ymin, ymax, class_name):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        self.class_name = class_name
        self.class_id = CLASSES.get(class_name, None)

    def draw_bounding_box(self, draw_image):
        draw_image.rectangle([(self.xmin, self.ymin), (self.xmax, self.ymax)],
                             outline=(255, 0, 0), width=5)
        draw_image.text((self.xmin, self.ymin - 10),
                        self.class_name)
