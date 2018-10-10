import os

from PIL import Image, ImageDraw, ImageFont

from utils.annotations import AnnotationParser


class Visualization:

    def __init__(self, image_path, annotation_path):
        self.image_path = image_path
        self.annotation_path = annotation_path

        self.image = Image.open(image_path)
        self.bounding_boxes = AnnotationParser(annotation_path).parse_annotations()

    def visualize_annotated_image(self):
        draw = ImageDraw.Draw(self.image)
        for bounding_box in self.bounding_boxes:
            bounding_box.draw_bounding_box(draw)
        self.image.show()