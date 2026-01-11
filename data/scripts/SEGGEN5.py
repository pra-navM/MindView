import os
import torch
import nibabel as nib
import numpy as np
import scipy.ndimage as ndimage
from monai.networks.nets import SegResNet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    NormalizeIntensityd, Orientationd, Spacingd, 
    Invertd, AsDiscreted, KeepLargestConnectedComponentd
)

def generate_segmentation_aligned(t1, t1ce, t2, flair, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Model
    model = SegResNet(
        blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1],
        init_filters=16, in_channels=4, out_channels=3
    ).to(device)
    
    weights_path = "./monai_brats_bundle/brats_mri_segmentation/models/model.pt"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # 2. Pre-processing
    data = {"image": [flair, t1ce, t1, t2]}
    pre_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])
    
    processed = pre_transforms(data)
    input_tensor = processed["image"].unsqueeze(0).to(device)

    # 3. Inference
    print("Running 3D sliding window inference...")
    with torch.no_grad():
        output = sliding_window_inference(
            inputs=input_tensor, 
            roi_size=(128, 128, 128), 
            sw_batch_size=1, 
            predictor=model, 
            overlap=0.7 # High overlap for smoother results
        )
    
    # 4. Inversion
    processed["pred"] = output[0] 
    inverter = Invertd(
        keys="pred",
        transform=pre_transforms,
        orig_keys="image",
        meta_key_postfix="meta_dict",
        nearest_interp=True,
        to_tensor=True,
    )
    inverted_dict = inverter(processed)
    
    # 5. Post-processing: Advanced Noise & Spottiness Control
    # Use higher thresholds for TC (Channel 0) to shrink the necrotic core
    # TC uses 0.7 instead of 0.5 to match "tiny" gold standard cores
    pred = inverted_dict["pred"].sigmoid()
    
    tc_mask = (pred[0] > 0.7).cpu().numpy() # Tumor Core (Shrunken)
    wt_mask = (pred[1] > 0.5).cpu().numpy() # Whole Tumor
    et_mask = (pred[2] > 0.5).cpu().numpy() # Enhancing Tumor

    # Fill small holes in the masks (fixes "spottiness" and "hollowness")
    def clean_mask(mask):
        # Fill holes
        mask = ndimage.binary_fill_holes(mask)
        # Remove small specks (Connected Components)
        labeled, num_features = ndimage.label(mask)
        if num_features > 0:
            sizes = ndimage.sum(mask, labeled, range(num_features + 1))
            max_label = np.argmax(sizes)
            mask = labeled == max_label
        return mask

    print("Cleaning and refining necrotic masks...")
    tc_mask = clean_mask(tc_mask)
    wt_mask = clean_mask(wt_mask)
    et_mask = clean_mask(et_mask)

    # 6. Definitive Label Mapping (Nested Logic)
    # Label 1: Necrotic (TC minus ET)
    # Label 2: Edema (WT minus TC)
    # Label 4: Enhancing (ET)
    final_seg = np.zeros(tc_mask.shape, dtype=np.uint8)
    
    final_seg[wt_mask] = 2  # Set everything to Edema
    final_seg[tc_mask] = 1  # Overlay the Core (Necrotic)
    final_seg[et_mask] = 4  # Overlay the Enhancing ring

    # 7. Save using ORIGINAL T1ce Header
    ref_img = nib.load(t1ce)
    new_img = nib.Nifti1Image(final_seg, ref_img.affine, ref_img.header)
    nib.save(new_img, output_path)
    
    print(f"SUCCESS: Refined segmentation saved to {output_path}")

if __name__ == "__main__":
    files = {
        "t1": "BraTS20_Training_368_t1.nii",
        "t1ce": "BraTS20_Training_368_t1ce.nii",
        "t2": "BraTS20_Training_368_t2.nii",
        "flair": "BraTS20_Training_368_flair.nii",
        "output_path": "BraTS20_368_seg_aligned.nii"
    }
    generate_segmentation_aligned(**files)