'''
Requires a folder in root directory called examples_root
Format should be many subfolders containing .jpg inside
'''

import os
import shutil
from sklearn.model_selection import train_test_split

# Given a main folder, extract all .jpg to a destination folder
# Works on the format we were given
def extract_images(source, dest):
    
    # Create the destination folder if it doesn't exist
    if not os.path.exists(dest):
        os.makedirs(dest)

    # Traverse through the source folder and its subfolders
    for foldername, subfolders, filenames in os.walk(source):
        for filename in filenames:
            if filename.lower().endswith('.jpg'):
                # Full path of the source image file
                source_image_path = os.path.join(foldername, filename)

                # Full path for the destination image file
                destination_image_path = os.path.join(dest, filename)

                # Copy the image to the destination folder
                shutil.move(source_image_path, destination_image_path)


def split_images(input_folder, train_folder, test_folder):
    
    # Create output folders if they don't exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # List all images in the input folder
    images = [f for f in os.listdir(input_folder) if f.endswith(('.jpg'))]

    # Split the images into training and test sets
    train_images, test_images = train_test_split(images, test_size=0.2, train_size = 0.8, random_state=42)

    # Move images to the train and test folders
    for image in train_images:
        shutil.move(os.path.join(input_folder, image), os.path.join(train_folder, image))

    for image in test_images:
        shutil.move(os.path.join(input_folder, image), os.path.join(test_folder, image))



def organize_images(input_folder, output_folder):
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Key to map image name to its class:
    '''
    back: violin_back
    bb: back_zoom
    bside: violin_left
    fb: front_zoom
    front: scroll_front
    head: scroll_left
    label: label
    rear: scroll_back
    top: violin_front
    treb: scroll_right
    tside: violin_right
    '''
    
    # Iterate through the images in the input folder
    for image_file in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_file)

        # Extract the class from the image name
        if "back" in image_file: 
            class_name = "violin_back"
        elif "bb" in image_file:
            class_name = "back_zoom"
        elif "bside" in image_file:
            class_name = "violin_left"
        elif "fb" in image_file:
            class_name = "front_zoom"
        elif "front" in image_file:
            class_name = "scroll_front"
        elif "head" in image_file:
            class_name = "scroll_left"
        elif "label" in image_file:
            class_name = "label"
        elif "rear" in image_file:
            class_name = "scroll_back"
        elif "top" in image_file:
            class_name = "violin_front"
        elif "treb" in image_file:
            class_name = "scroll_right"
        elif "tside" in image_file:
            class_name = "violin_right"
        else:
            class_name = "other"

        # Create a subfolder for the class if it doesn't exist
        class_folder = os.path.join(output_folder, class_name)
        os.makedirs(class_folder, exist_ok=True)

        # Move the image to the appropriate subfolder
        shutil.move(image_path, os.path.join(class_folder, image_file))

def remove_excess():
    os.rmdir(r'Image_examples')
    shutil.rmtree(r'violin_data\test\other')
    shutil.rmtree(r'violin_data\train\other')


source_folder = r'examples_root'
dest_folder = r'Image_examples'

input_data_folder = r'Image_examples'
train_folder_path = r'violin_data\train'
test_folder_path = r'violin_data\test'

extract_images(source_folder, dest_folder)

split_images(input_data_folder, train_folder_path, test_folder_path)

organize_images(r'violin_data\train', r'violin_data\train')
organize_images(r'violin_data\test', r'violin_data\test')

remove_excess()
