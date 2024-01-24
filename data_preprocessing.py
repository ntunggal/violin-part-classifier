import os
import shutil
from sklearn.model_selection import train_test_split

def split_images(input_folder, train_folder, test_folder):
    # Create output folders if they don't exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # List all images in the input folder
    images = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

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

    class_info = ['label', 'bside', 'tside', 'fb', 'front', 'head', 'rear', 'treb', 'top']
    
    #key to map image name to its class:
    '''
    label = label
    front, head, rear, treb = scroll
    bside, tside = side_view
    fb = f-holes
    top = full_view
    bb = back_zoomed
    back = back_view
    '''
    
    # Iterate through the images in the input folder
    for image_file in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_file)


        # Extract the class from the image name
        if "front" in image_file or "head" in image_file or "rear" in image_file or "treb" in image_file: 
            class_name = "scroll"
        elif "bside" in image_file or "tside" in image_file:
            class_name = "side_view"
        elif "fb" in image_file:
            class_name = "f-holes"
        elif "top" in image_file:
            class_name = "full_view"
        elif "bb" in image_file:
            class_name = "back_zoomed"
        elif "back" in image_file:
            class_name = "back_view"
        else:
            class_name = "other"

        # Create a subfolder for the class if it doesn't exist
        class_folder = os.path.join(output_folder, class_name)
        os.makedirs(class_folder, exist_ok=True)

        # Move the image to the appropriate subfolder
        shutil.move(image_path, os.path.join(class_folder, image_file))

input_data_folder = r'Image_examples'
train_folder_path = r'dataset\train'
test_folder_path = r'dataset\test'

split_images(input_data_folder, train_folder_path, test_folder_path)

organize_images(r'dataset\train', r'dataset\train')
organize_images(r'dataset\test', r'dataset\test')