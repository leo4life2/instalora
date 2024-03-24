import shutil
import os
import modal
from modal import Stub, Volume

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

stub = Stub(name="kohya-app")

@stub.function(mounts=[modal.Mount.from_local_dir("/Users/leoli/Desktop/ivyhacks/data/leo/leo", remote_path="/root/data/leo")], volumes=VOLUME_CONFIG)
def upload():
    source_dir = "/root/data/leo"
    target_dir = "/training_data/leo"

    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Copy the content of source_dir to target_dir
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)  # dirs_exist_ok=True allows for the target directory to already exist
    
    training_data_volume.commit()
    
@stub.function(volumes=VOLUME_CONFIG)
def move():
    # Define the directory and file paths
    new_dir_path = '/model/loras'
    file_to_move = '/model/new_lora.safetensors'
    new_file_path = '/model/loras/leo.safetensors'

    # Create the new directory if it doesn't exist
    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)

    # Move the file and rename it
    shutil.move(file_to_move, new_file_path)
    
    model_volume.commit()

@stub.local_entrypoint()
def run():
    # upload.remote()
    move.remote()