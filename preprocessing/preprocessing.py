# preprocessing.py
from zenml.steps import step
import os
import pandas as pd
import xml.etree.ElementTree as xet
from shutil import copy

@step
def preprocess_data(df: pd.DataFrame, output_folder: str = './yolov5/data_images/train') -> None:
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    def parsing(path):
        parser = xet.parse(path).getroot()
        parser_size = parser.find('size')
        width = int(parser_size.find('width').text)
        height = int(parser_size.find('height').text)
        return width, height

    # Parse data and create training labels
    values = df[['filename', 'xmin', 'xmax', 'ymin', 'ymax']].values
    for fname, xmin, xmax, ymin, ymax in values:
        image_name = os.path.split(fname)[-1]
        txt_name = os.path.splitext(image_name)[0]

        dst_image_path = os.path.join(output_folder, image_name)
        dst_label_file = os.path.join(output_folder, txt_name + '.txt')

        # Copy image
        copy(fname, dst_image_path)

        # Write YOLO-style label (assuming a single class "0")
        width, height = parsing(fname)
        center_x = (xmax + xmin) / (2 * width)
        center_y = (ymax + ymin) / (2 * height)
        bb_width = (xmax - xmin) / width
        bb_height = (ymax - ymin) / height

        with open(dst_label_file, 'w') as f:
            f.write(f"0 {center_x} {center_y} {bb_width} {bb_height}\n")
