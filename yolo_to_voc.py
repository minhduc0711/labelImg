import os
import argparse

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


from libs.labelFile import LabelFile
from libs.shape import Shape
from libs.yolo_io import YoloReader


# def loadYOLOTXTByFilename(self, txtPath):
    # print(shapes)
    # self.loadLabels(shapes)
    # self.canvas.verified = tYoloParseReader.verified


def read(filename, default=None):
    try:
        with open(filename, 'rb') as f:
            return f.read()
    except:
        return default


def format_shape(s):
    return dict(label=s.label,
                line_color=s.line_color.getRgb(),
                fill_color=s.fill_color.getRgb(),
                points=[(p.x(), p.y()) for p in s.points],
                # add chris
                difficult=s.difficult)


def loadLabels(shapes):
    s = []
    for label, points, line_color, fill_color, difficult in shapes:
        shape = Shape(label=label)
        for x, y in points:
            # Ensure the labels are within the bounds of the image. If not, fix them.
            # x, y, snapped = self.canvas.snapPointToCanvas(x, y)
            # if snapped:
                # self.setDirty()

            shape.addPoint(QPointF(x, y))
        shape.difficult = difficult
        shape.close()
        s.append(shape)

    return s


def main():
    for fname in os.listdir(args.dir):
        root, ext = os.path.splitext(fname)
        if ext != ".txt" or root == "classes":
            continue

        image_path = os.path.join(args.dir, f"{root}.jpg")
        image_data = read(image_path)
        image = QImage.fromData(image_data)

        yolo_path = os.path.join(args.dir, fname)
        tYoloParseReader = YoloReader(yolo_path, image)

        shapes = tYoloParseReader.getShapes()
        shapes = loadLabels(shapes)
        shapes = [format_shape(shape) for shape in shapes]

        label_file = LabelFile()
        voc_path = os.path.join(args.dir, f"{root}.xml")
        label_file.savePascalVocFormat(
            voc_path, shapes, image_path, image_data)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("dir", help="path to the image folder",
                    type=str)
    args = ap.parse_args()
    args.dir = os.path.abspath(args.dir)
    print(args.__dict__)

    main()
