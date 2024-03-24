import os
from PIL import Image
from multiprocessing import Pool, cpu_count

def process_image(data):
    filename, directory, output_dir, text_content, side_length = data
    # Open the image
    with Image.open(os.path.join(directory, filename)) as img:
        # Crop to a center square if needed
        if img.width != img.height:
            new_size = min(img.width, img.height)
            left = (img.width - new_size) / 2
            top = (img.height - new_size) / 2
            img = img.crop((left, top, left + new_size, top + new_size))
        
        # Resize the image
        img = img.resize((side_length, side_length), Image.Resampling.LANCZOS)

        # Save the processed image
        img.save(os.path.join(output_dir, filename))
    
    # Create the corresponding text file
    text_file_path = os.path.join(output_dir, f"{filename[:-4]}.txt")
    with open(text_file_path, 'w') as text_file:
        text_file.write(text_content)

def process_images(input_dir, text_content, side_length):
    # Create the output directory
    output_dir = os.path.join(input_dir, '../preprocessed')
    os.makedirs(output_dir, exist_ok=True)

    # Get all .png files in the input directory
    files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    
    # Prepare data for multiprocessing
    data = [(filename, input_dir, output_dir, text_content, side_length) for filename in files]
    
    # Process images in parallel
    with Pool(min(4, cpu_count())) as pool:
        pool.map(process_image, data)

if __name__ == '__main__':
    # Replace these with your actual paths and values
    input_directory = './data/leo/images'
    user_text_content = 'l3o, person'
    square_side_length = 768

    process_images(input_directory, user_text_content, square_side_length)