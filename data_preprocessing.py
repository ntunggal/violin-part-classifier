import os
import shutil
from sklearn.model_selection import train_test_split

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

    class_info = ['back', 'bb', 'bside', 'fb', 'front', 'head', 'label', 'rear', 'top', 'treb', 'tside']
    
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

input_data_folder = r'Image_examples'
train_folder_path = r'violin_data\train'
test_folder_path = r'violin_data\test'

split_images(input_data_folder, train_folder_path, test_folder_path)

organize_images(r'violin_data\train', r'violin_data\train')
organize_images(r'violin_data\test', r'violin_data\test')