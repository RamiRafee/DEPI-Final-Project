import os
import pandas as pd
import xml.etree.ElementTree as xet
from shutil import copy
import mlflow

def preprocess_data(df: pd.DataFrame, output_folder: str = './yolov5/data_images/train') -> None:
    """Preprocess the data and create YOLO-style labels from XML annotations."""
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    def parsing(path):
        """Parse the XML file to extract width and height."""
        parser = xet.parse(path).getroot()
        parser_size = parser.find('size')
        width = int(parser_size.find('width').text)
        height = int(parser_size.find('height').text)
        return width, height

    # Iterate through the dataframe to create labels and copy images
    for _, row in df.iterrows():
        # XML file path
        xml_fname = row['filepath']
        
        # Generate the corresponding image file name (adjust the extension as needed)
        image_fname = os.path.splitext(xml_fname)[0] + '.jpg'  # Change .jpg to the actual image format if different
        xmin = row['xmin']
        xmax = row['xmax']
        ymin = row['ymin']
        ymax = row['ymax']

        # Extract image name and label file name
        image_name = os.path.split(image_fname)[-1]
        txt_name = os.path.splitext(image_name)[0]

        dst_image_path = os.path.join(output_folder, image_name)
        dst_label_file = os.path.join(output_folder, txt_name + '.txt')

        try:
            # Check if the image already exists in the destination
            if not os.path.exists(dst_image_path):
                # Copy the image only if it doesn't already exist
                copy(image_fname, dst_image_path)
                print(f"Copied image: {image_fname} to {dst_image_path}")
            else:
                print(f"Image already exists: {dst_image_path}. Skipping copy.")
        except Exception as e:
            print(f"Failed to copy image {image_fname}: {e}")

        # Write YOLO-style label (assuming a single class "0")
        width, height = parsing(xml_fname)
        center_x = (xmax + xmin) / (2 * width)
        center_y = (ymax + ymin) / (2 * height)
        bb_width = (xmax - xmin) / width
        bb_height = (ymax - ymin) / height

        with open(dst_label_file, 'w') as f:
            f.write(f"0 {center_x} {center_y} {bb_width} {bb_height}\n")

    # Log preprocessing information
    mlflow.log_param("output_folder", output_folder)
    mlflow.log_param("num_samples", len(df))
