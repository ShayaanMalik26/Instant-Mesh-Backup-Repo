import os
import numpy as np

def save_npz_for_each_folder(root_dir):
    # Walk through the directory
    for subdir, dirs, files in os.walk(root_dir):
        # Filter out only the npy files
        npy_files = [file for file in files if file.endswith('.npy')]
        
        if npy_files:
            # Create a dictionary to hold the npy data
            cam_poses = []

            for npy_file in npy_files:
                # Load the npy file
                data = np.load(os.path.join(subdir, npy_file))
                # Store it in the dictionary using the file name as the key
                cam_poses.append(data)
            
            # Determine the output npz file path
            npz_file_name = "cameras.npz"
            npz_file_path = os.path.join(subdir, npz_file_name)
            
            # Save the dictionary to the npz file
            np.savez(npz_file_path, cam_poses=cam_poses)
            print(f"Saved: {npz_file_path}")

# Replace this with the actual path to your rendering_zero123plus directory
root_directory = "/fsx/ubuntu/3d_model_finetuning/InstantMesh/data/custom/rendering_zero123plus"
save_npz_for_each_folder(root_directory)
