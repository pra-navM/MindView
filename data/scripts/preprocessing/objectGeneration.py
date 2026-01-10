import nibabel as nib
import numpy as np
from skimage import measure

def convert_nifti_to_obj(nifti_path, output_obj_path):
    try:
        # 1. Load the NIfTI file
        img = nib.load(nifti_path)
        data = img.get_fdata()
        
        # --- DATA CHECK ---
        val_min = np.min(data)
        val_max = np.max(data)
        print(f"File: {nifti_path}")
        print(f"Data range: {val_min} to {val_max}")

        if val_max <= 0.5:
            print("❌ Error: This mask is empty or contains no brain data. Skipping...")
            return
        # ------------------

        # 2. Apply Marching Cubes
        verts, faces, normals, values = measure.marching_cubes(data, level=0.5)

        # 3. Save as .obj file
        with open(output_obj_path, 'w') as f:
            f.write("# OBJ file generated from NIfTI\n")
            for v in verts:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        print(f"✅ Success! Model saved as: {output_obj_path}")
    except Exception as e:
        print(f"Error during conversion: {e}")

# --- EXECUTION ---
# If s0011 fails, try s0012 or another folder in your dataset
file_to_process = r'C:\Users\prana\Downloads\Totalsegmentator_dataset_small_v201\s0511\segmentations\brain.nii.gz'
convert_nifti_to_obj(file_to_process, 'brain_model.obj')