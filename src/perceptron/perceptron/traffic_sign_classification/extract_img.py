import pickle
import os
import cv2
import numpy as np

def extract_img_from_p(pickle_file, output_folder):
    # Load the pickle file
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate through the data and extract images
    for i, item in enumerate(data['features']):
        # Convert the img data to a numpy array
        img_array = np.array(item, dtype=np.uint8)
        # Save the img using OpenCV
        output_path = os.path.join(output_folder, f'{i}.png')
        cv2.imwrite(output_path, img_array)

if __name__ == "__main__":
    pickle_file = 'test.p'
    output_folder = pickle_file.split('.')[0]
    extract_img_from_p(pickle_file, output_folder)