import nibabel as nib
import numpy as np
from skimage import measure

def create_classified_tumor_obj(brain_path, seg_path, output_path):
    # 1. Load Data
    brain_img = nib.load(brain_path)
    seg_img = nib.load(seg_path)
    
    brain_data = brain_img.get_fdata()
    seg_data = seg_img.get_fdata()
    affine = brain_img.affine

    # 2. Define labels for classification
    # Note: Label 3 is typically unused in BraTS20
    regions = {
        "BrainShell": (brain_data > np.percentile(brain_data, 10)).astype(float),
        "NecroticCore": (seg_data == 1).astype(float),
        "Edema": (seg_data == 2).astype(float),
        "EnhancingTumor": (seg_data == 4).astype(float)
    }

    def transform(v, aff):
        v = np.c_[v, np.ones(v.shape[0])]
        return (v @ aff.T)[:, :3]

    all_verts = []
    all_faces = []
    current_offset = 0

    # 3. Process each region
    with open(output_path, 'w') as f:
        f.write("# Classified Brain MRI Mesh\n")
        
        for name, mask in regions.items():
            # Check if the region actually has voxels (some tumors lack certain parts)
            if np.sum(mask) == 0:
                print(f"Skipping {name}: No voxels found.")
                continue

            # Generate mesh for this specific label
            try:
                verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
                verts = transform(verts, affine)

                # Write group header
                f.write(f"g {name}\n")
                
                # Write vertices
                for v in verts:
                    f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
                
                # Write faces with the running offset
                for face in faces:
                    f.write(f"f {face[0]+1+current_offset} {face[1]+1+current_offset} {face[2]+1+current_offset}\n")
                
                current_offset += len(verts)
                print(f"Added {name} to mesh.")
            except RuntimeError:
                print(f"Skipping {name}: Surface could not be computed.")

    print(f"\nSuccessfully saved classified mesh to: {output_path}")

# Configuration
brain_input = 'BraTS20_Training_368_t1.nii'
seg_input = 'BraTS20_368_seg_aligned.nii'
output_file = 'BraTS20_368_Full_Classification4.obj'

create_classified_tumor_obj(brain_input, seg_input, output_file)