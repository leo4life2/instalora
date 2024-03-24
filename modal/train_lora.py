import os
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, HTTPException
from modal import (
    Image,
    Stub,
    Volume,
    asgi_app,
    enter,
    method,
)

web_app = FastAPI()
web_image = Image.debian_slim().pip_install("gradio~=3.50.2", "pillow~=10.2.0")

image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "accelerate==0.25.0",
        "transformers==4.36.2",
        "ftfy==6.1.1",
        "opencv-python==4.7.0.68",
        "einops==0.7.0",
        "pytorch-lightning==1.9.0",
        "tensorboard==2.10.1",
        "safetensors==0.4.2",
        "altair==4.2.2",
        "easygui==0.98.3",
        "toml==0.10.2",
        "voluptuous==0.13.1",
        "huggingface-hub==0.20.1",
        "open-clip-torch==2.20.0",
        "rich==13.7.0",
        "gradio~=3.50.2",
        "torch",
        "torchvision",
        "xformers",
        "requests",
        "bitsandbytes",
        "diffusers",
        "omegaconf",
        "safetensors"
    )
    .apt_install(
        "build-essential",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgl1-mesa-glx",
        "ffmpeg",
        "libjpeg-dev",
        "libpng-dev",
        "git"
    )

    # Perform a shallow fetch of just the target `diffusers` commit, checking out
    # the commit in the container's current working directory, /root.
    .run_commands(
        # "cd /root && git init .",
        # "cd /root && git remote add origin https://github.com/huggingface/diffusers",
        # f"cd /root && git fetch --depth=1 origin {GIT_SHA} && git checkout {GIT_SHA}",
        # "pip install git+https://github.com/huggingface/diffusers.git",
        "cd /root && git clone https://github.com/kohya-ss/sd-scripts.git"
    )
)

from PIL import Image

with image.imports():
    # we import these dependencies only inside the container
    import diffusers
    import huggingface_hub
    import torch
    import io

# ## Set up `Volume`s for training data and model output
#
# Modal can't access your local filesystem, so you should set up a `Volume` to eventually save the model once training is finished.

stub = Stub(name="kohya-app")

MODEL_DIR = Path("/model")
training_data_volume = Volume.from_name(
    "diffusers-training-data-volume", create_if_missing=True
)
model_volume = Volume.from_name(
    "diffusers-model-volume", create_if_missing=True
)

VOLUME_CONFIG = {
    "/training_data": training_data_volume,
    "/model": model_volume,
}

def extract_frames(video_path, output_dir, lora_name):
    """
    Extract frames from a video using ffmpeg, saving them in a specified directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        'ffmpeg', '-i', video_path, '-vf', 'fps=15',
        os.path.join(output_dir, f"{lora_name}_%04d.png")
    ]
    
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def process_image(data):
    filename, directory, output_dir, text_content, side_length = data
    with Image.open(os.path.join(directory, filename)) as img:
        if img.width != img.height:
            new_size = min(img.width, img.height)
            left = (img.width - new_size) / 2
            top = (img.height - new_size) / 2
            img = img.crop((left, top, left + new_size, top + new_size))
        
        img = img.resize((side_length, side_length), Image.Resampling.LANCZOS)

        img.save(os.path.join(output_dir, filename))
    
    text_file_path = os.path.join(output_dir, f"{filename[:-4]}.txt")
    with open(text_file_path, 'w') as text_file:
        text_file.write(text_content)

def process_images(input_dir, text_content, side_length):
    output_dir = os.path.join('/training_data', os.path.basename(input_dir))
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    data = [(filename, input_dir, output_dir, text_content, side_length) for filename in files]

    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count())) as executor:
        future_to_image = {executor.submit(process_image, d): d for d in data}
        for future in as_completed(future_to_image):
            try:
                future.result()  # You can handle exceptions here if you wish
            except Exception as exc:
                print(f'Generated an exception: {exc}')

    print("end of process images", os.listdir(output_dir))
    training_data_volume.commit()

@stub.function(cpu=8.0, image=image, volumes=VOLUME_CONFIG, timeout=60 * 60 * 2)
def process_videos(video1_path, video2_path, lora_name, caption):

    temp_dir = os.path.join('./temp', lora_name)
    os.makedirs(temp_dir, exist_ok=True)
    
    # check if video1_path and video2_path exist
    if not os.path.exists(video1_path):
        raise FileNotFoundError(f"File {video1_path} does not exist.")
    if not os.path.exists(video2_path):
        raise FileNotFoundError(f"File {video2_path} does not exist.")
    
    extract_frames(video1_path, temp_dir, lora_name + "_1")
    extract_frames(video2_path, temp_dir, lora_name + "_2")
    
    process_images(temp_dir, caption, 768)
    training_data_volume.commit()

@stub.function(
    image=image,
    cpu=8.0,
    gpu="H100",
    volumes=VOLUME_CONFIG,
    _allow_background_volume_commits=True,  # enables saving of larger files on Modal Volumes
    timeout=60 * 60 * 2  # two hours, for longer training jobs
)
# ## Define the training function
# Now, finally, we define the training function itself.
# This training function does a bunch of preparatory things, but the core of the work is in the training script.
# Depending on which Diffusers script you are using, you will want to modify the script name, and the arguments that are passed to it.
def train(lora_name):
    import requests
    import toml
    
    if not os.path.isdir(f"/training_data/{lora_name}"):
        raise FileNotFoundError(f"Directory /training_data/{lora_name} does not exist.")

    # 1. download realisticvision.safetensors into /model if it doesn't exist at https://civitai.com/api/download/models/130072?type=Model&format=SafeTensor&size=full&fp=fp16
    # Define the model file path and URL
    model_dir = "/model"
    model_filename = "realisticvision.safetensors"
    model_path = os.path.join(model_dir, model_filename)
    url = "https://civitai.com/api/download/models/130072?type=Model&format=SafeTensor&size=full&fp=fp16"

    # Check if the model file already exists
    if not os.path.exists(model_path):
        # Ensure the model directory exists
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Download the model
        print(f"Downloading {model_filename}...")
        response = requests.get(url, allow_redirects=True)

        # Check if the request was successful
        if response.status_code == 200:
            # Write the content to the file
            with open(model_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded and saved to {model_path}")
        else:
            print(f"Failed to download the model. Status code: {response.status_code}")
    else:
        print(f"{model_filename} already exists in {model_dir}")
    
    # Step 2: Create lora.toml with default content if it doesn't exist
    lora_file_path = '/model/lora.toml'
    if not os.path.exists(lora_file_path):
        default_content = '''
    [general]
    enable_bucket = true

    [[datasets]]
    resolution = 768
    batch_size = 16

    [[datasets.subsets]]
    image_dir = 'DNE'
    caption_extension = '.txt'
    num_repeats = 10
    '''
        with open(lora_file_path, 'w') as file:
            file.write(default_content)

    # Step 3: Modify the image_dir in lora.toml
    with open(lora_file_path, 'r') as file:
        lora_config = toml.load(file)

    lora_config['datasets'][0]['subsets'][0]['image_dir'] = f'/training_data/{lora_name}'

    with open(lora_file_path, 'w') as file:
        toml.dump(lora_config, file)
    
    # 4. execute the training script with the appropriate arguments
    def _exec_subprocess(cmd: list[str]):
        """Executes subprocess and prints log to terminal while subprocess is running."""
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        with process.stdout as pipe:
            for line in iter(pipe.readline, b""):
                line_str = line.decode()
                print(f"{line_str}", end="")

        if exitcode := process.wait() != 0:
            raise subprocess.CalledProcessError(exitcode, "\n".join(cmd))

    print("launching training script")
    _exec_subprocess(
        [
            "accelerate",
            "launch",
            "--num_cpu_threads_per_process", "8",
            "sd-scripts/train_network.py",
            "--pretrained_model_name_or_path=/model/realisticvision.safetensors",
            "--dataset_config=/model/lora.toml",
            "--output_dir=/model/loras",
            f"--output_name={lora_name}",
            "--save_model_as=safetensors",
            "--prior_loss_weight=1.0",
            "--max_train_steps=400",
            "--learning_rate=1e-4",
            "--optimizer_type=AdamW8bit",
            "--xformers",
            "--mixed_precision=fp16",
            "--cache_latents",
            "--save_every_n_epochs=1",
            "--network_module=networks.lora",
        ]
    )

    # The trained model artefacts have been output to the volume mounted at `MODEL_DIR`.
    # To persist these artefacts for use in future inference function calls, we 'commit' the changes
    # to the volume.
    model_volume.commit()

@stub.function(
    image=image,
    gpu="T4",
    volumes=VOLUME_CONFIG,
    _allow_background_volume_commits=True,  # enables saving of larger files on Modal Volumes
    timeout=60 * 60 * 2  # two hours, for longer training jobs
)
def test():
    diffusers.StableDiffusionPipeline.from_pretrained(
        "/model/dreamshaper.safetensors", torch_dtype=torch.float16
    ).to("cuda")
    print("Testing Stable Diffusion Pipeline")
    

@stub.local_entrypoint()
def run():
    test.remote()
    # train.remote("leo")

@stub.cls(gpu="H100",
          volumes=VOLUME_CONFIG,
          image=image)
class StableDiffusionLoRA:    
    base_path = "/model/dreamshaper.safetensors"

    @enter()  # when a new container starts, we load the base model into the GPU
    def load(self):
        # show all contents of /model
        print("check /model")
        print(os.listdir("/model"))
        
        self.pipe = diffusers.StableDiffusionPipeline.from_single_file(
            self.base_path, torch_dtype=torch.float16
        ).to("cuda")

    @method()  # at inference time, we pull in the LoRA weights and pass the final model the prompt
    def run_inference_with_lora(
        self, lora_name: str, prompt: str, negative_prompt, width=768, height=768, num_steps=20, guidance_scale=7, num_images_per_prompt=1, seed: int = -1
    ) -> bytes:
        # print all parameters out
        print(f"lora_name: {lora_name}")
        print(f"prompt: {prompt}")
        print(f"negative_prompt: {negative_prompt}")
        print(f"width: {width}")
        print(f"height: {height}")
        print(f"num_steps: {num_steps}")
        print(f"guidance_scale: {guidance_scale}")
        print(f"num_images_per_prompt: {num_images_per_prompt}")
        print(f"seed: {seed}")
        seed = seed if seed != -1 else torch.randint(0, 1000000, (1,)).item()
        
        self.pipe.load_lora_weights("/model/loras", weight_name=f"{lora_name}.safetensors")
        
        def sc(self, clip_input, images) : return images, [False for i in images]
        diffusers.pipelines.stable_diffusion.safety_checker.StableDiffusionSafetyChecker.forward = sc
        
        images = self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            width=int(width),
            height=int(height),
            num_images_per_prompt=int(num_images_per_prompt),
            guidance_scale=guidance_scale,
            num_inference_steps=int(num_steps),
            generator=torch.manual_seed(seed),
        ).images

        images_bytes = []
        for img in images:
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            images_bytes.append(buffer.getvalue())

        return images_bytes
        
        # image = self.pipe(
        #     prompt,
        #     negative_prompt=negative_prompt,
        #     width=int(width),
        #     height=int(height),
        #     num_images_per_prompt=int(num_images_per_prompt),
        #     guidance_scale=guidance_scale,
        #     num_inference_steps=int(num_steps),
        #     generator=torch.manual_seed(seed),
        # ).images[0]

        # buffer = io.BytesIO()
        # image.save(buffer, format="PNG")

        # return buffer.getvalue()

@stub.function(image=web_image, keep_warm=1, container_idle_timeout=60 * 20, volumes=VOLUME_CONFIG)
@asgi_app()
def app():
    import gradio as gr
    from gradio.routes import mount_gradio_app
    from PIL import Image
    import io
    import os
    import shutil
    
    """A simple Gradio interface for training and inference."""

    def get_lora_models():
        """Lists all available LoRA models from /model/loras."""
        loras_path = "/model/loras"
        try:
            return [f.replace('.safetensors', '') for f in os.listdir(loras_path) if os.path.isfile(os.path.join(loras_path, f))]
        except Exception as e:
            print(f"Error accessing LoRA models directory: {e}")
            return []
    
    def refresh_models():
        return get_lora_models()
    
    def save_uploaded_file(uploaded_file):
        """
        Saves the uploaded file to a temporary file and returns the path.
        """
        # get file name, base path
        filename = os.path.basename(uploaded_file.name)
        if not os.path.exists("/training_data/tmp"):
            os.makedirs("/training_data/tmp", exist_ok=True)
        shutil.copyfile(uploaded_file.name, f"/training_data/tmp/{filename}")
        training_data_volume.commit()
        return f"/training_data/tmp/{filename}"
    
    def train_interface(lora_name, captions, closeup_video, far_video):
        # if videos exist
        print("closeup video name", closeup_video.name)
        if not os.path.exists(closeup_video.name):
            return f"File {closeup_video} does not exist."
        if not os.path.exists(far_video.name):
            return f"File {far_video} does not exist." 
        
        closeup_video_path = save_uploaded_file(closeup_video)
        far_video_path = save_uploaded_file(far_video)
        
        process_videos.remote(closeup_video_path, far_video_path, lora_name, captions)
        # image caption pairs should be at /training_data/{lora_name}
        train.remote(lora_name)
        return "Done"

    def inference_interface(text_lora_id, lora_id, prompt, negative_prompt, width, height, num_images_per_prompt, guidance_scale, num_inference_steps, seed=-1):
        # wipe everything in /tmp/gradio
        if os.path.exists("/tmp/gradio"):
            shutil.rmtree("/tmp/gradio")
        
        lora_id = text_lora_id if text_lora_id else lora_id
        try:
            images_bytes = StableDiffusionLoRA().run_inference_with_lora.remote(
                lora_id, prompt, negative_prompt, width, height, num_inference_steps, guidance_scale, num_images_per_prompt, seed
            )

            # Convert bytes back to PIL Images for Gradio gallery
            images = [Image.open(io.BytesIO(img_bytes)) for img_bytes in images_bytes]

            return images
        
            # return Image.open(
            #     io.BytesIO(
            #         StableDiffusionLoRA().run_inference_with_lora.remote(
            #             lora_id, prompt, negative_prompt, width, height, num_inference_steps, guidance_scale, num_images_per_prompt, seed
            #         )
            #     ),
            # )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    with gr.Blocks() as gr_interface:
        gr.Markdown("## InstaLoRA")
        with gr.Tab("Training"):
            with gr.Group():
                lora_name_input = gr.Textbox(label="Lora Name")
                captions_input = gr.Textbox(label="Captions")
                closeup_video_input = gr.File(label="Closeup Video", type="file")
                far_video_input = gr.File(label="Farther Out Video", type="file")
                train_button = gr.Button("Start Training")
            train_output = gr.Textbox(label="Training Output")
            train_button.click(train_interface, inputs=[lora_name_input, captions_input, closeup_video_input, far_video_input], outputs=train_output)

        with gr.Tab("Inference"):
            with gr.Group():
                refresh_button = gr.Button("Refresh Models")
                lora_id_input = gr.Dropdown(label="Select a LoRA Model", choices=get_lora_models())
                lora_model_text = gr.Textbox(label="Or, enter a LoRA Model name")
                refresh_button.click(fn=refresh_models, inputs=[], outputs=lora_id_input)
                prompt_input = gr.Textbox(label="Prompt")
                negative_prompt_input = gr.Textbox(label="Negative Prompt")
                width_input = gr.Number(label="Width", value=512)
                height_input = gr.Number(label="Height", value=512)
                num_images_per_prompt_input = gr.Number(label="Number of Images per Prompt", value=1)
                guidance_scale_input = gr.Number(label="Guidance Scale", value=7)
                num_inference_steps_input = gr.Number(label="Number of Inference Steps", value=50)
                seed_input = gr.Number(label="Seed", value=-1)
                infer_button = gr.Button("Generate Image")
            # image_output = gr.Image(label="Generated Image")
            image_output = gr.Gallery(label="Generated Image")
            infer_button.click(
                inference_interface,
                inputs=[
                    lora_model_text, lora_id_input, prompt_input, negative_prompt_input, width_input, height_input,
                    num_images_per_prompt_input, guidance_scale_input, num_inference_steps_input, seed_input
                ],
                outputs=image_output
            )

    return mount_gradio_app(app=web_app, blocks=gr_interface, path="/")