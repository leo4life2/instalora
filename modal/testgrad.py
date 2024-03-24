import gradio as gr
import numpy as np
from PIL import Image
import io

def generate_images():
    images = []  # This will store the PIL Images, not bytes
    for _ in range(4):
        # Generate a random color
        color = np.random.randint(0, 255, (3,))
        
        # Create an image with the random color
        image = np.ones((100, 100, 3), dtype=np.uint8)
        image[:, :] = color
        
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image)
        
        # Convert PIL Image to BytesIO and then back to PIL Image
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)  # Important: move back to the start of the BytesIO object
        pil_image = Image.open(img_byte_arr)
        
        # Append the PIL Image to the list
        images.append(pil_image)
    
    return images

iface = gr.Interface(
    fn=generate_images,
    inputs=None,
    outputs=gr.Gallery(label="Random Color Images"),
    examples=[[]]
)

iface.launch()
