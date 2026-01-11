"""MONAI SegResNet tumor inference wrapper for brain MRI segmentation.

This module requires multi-modal MRI input (T1, T1ce, T2, FLAIR) for accurate
tumor detection. For single-modality input, tumor detection is skipped.

The model outputs BraTS-style labels:
- 1: Necrotic/non-enhancing tumor core
- 2: Peritumoral edema
- 4: Enhancing tumor

These are remapped to our internal labels (100, 101, 102) to avoid conflicts
with SynthSeg anatomical labels.
"""
from pathlib import Path
from typing import Callable, Dict, Optional

import nibabel as nib
import numpy as np
import scipy.ndimage as ndimage

from config import settings
from services.model_manager import ModelManager, ModelLoadError, InferenceError
from services.synthseg_service import remap_brats_labels


# Required modalities for tumor detection
REQUIRED_MODALITIES = ["t1", "t1ce", "t2", "flair"]

# Model weights path (relative to backend directory)
MODEL_WEIGHTS_PATH = Path(__file__).parent.parent / "models" / "brats_segresnet.pt"


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Clean binary mask by filling holes and keeping largest connected component.
    From SEGGEN5.py - effective for BraTS tumor masks.
    """
    # Fill holes
    mask = ndimage.binary_fill_holes(mask)
    # Remove small specks (keep largest connected component)
    labeled, num_features = ndimage.label(mask)
    if num_features > 0:
        sizes = ndimage.sum(mask, labeled, range(num_features + 1))
        max_label = np.argmax(sizes)
        mask = labeled == max_label
    return mask


class TumorInference:
    """Wrapper for MONAI SegResNet tumor segmentation model."""

    _model = None
    _model_path: Optional[Path] = None
    _device = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if MONAI and PyTorch are available."""
        try:
            import torch
            import monai
            return True
        except ImportError:
            return False

    @classmethod
    def get_model_path(cls) -> Path:
        """Get the path to model weights."""
        # Check configured path first
        if MODEL_WEIGHTS_PATH.exists():
            return MODEL_WEIGHTS_PATH

        # Fallback to monai_brats_bundle location (relative to cwd)
        bundle_path = Path("./monai_brats_bundle/brats_mri_segmentation/models/model.pt")
        if bundle_path.exists():
            return bundle_path

        # Check at MindView project root (parent of apps/backend)
        project_root = Path(__file__).parent.parent.parent.parent
        root_bundle_path = project_root / "monai_brats_bundle" / "brats_mri_segmentation" / "models" / "model.pt"
        if root_bundle_path.exists():
            return root_bundle_path

        # Check in data/scripts location
        scripts_path = project_root / "data" / "scripts" / "monai_brats_bundle" / "brats_mri_segmentation" / "models" / "model.pt"
        if scripts_path.exists():
            return scripts_path

        raise ModelLoadError(
            f"Model weights not found. Please copy weights to {MODEL_WEIGHTS_PATH} "
            "or ensure monai_brats_bundle is available."
        )

    @classmethod
    def load_model(cls):
        """Load MONAI SegResNet model (lazy loading)."""
        if cls._model is not None:
            return cls._model

        if not cls.is_available():
            raise ModelLoadError("MONAI or PyTorch not installed")

        try:
            import torch
            from monai.networks.nets import SegResNet

            # Get model path
            model_path = cls.get_model_path()
            cls._model_path = model_path

            device = ModelManager.get_device()
            cls._device = device
            print(f"Loading MONAI SegResNet model from {model_path}...")

            # Initialize model architecture (matches SEGGEN5.py)
            model = SegResNet(
                blocks_down=[1, 2, 2, 4],
                blocks_up=[1, 1, 1],
                init_filters=16,
                in_channels=4,  # FLAIR, T1ce, T1, T2
                out_channels=3  # 3 tumor classes (TC, WT, ET)
            )

            # Load weights
            state_dict = torch.load(str(model_path), map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            cls._model = model
            print("MONAI SegResNet model loaded successfully")
            return cls._model

        except Exception as e:
            raise ModelLoadError(f"Failed to load MONAI model: {e}")

    @classmethod
    def check_multimodal_input(cls, input_paths: Dict[str, Path]) -> bool:
        """Check if all required modalities are available."""
        for modality in REQUIRED_MODALITIES:
            if modality not in input_paths:
                return False
            path = input_paths[modality]
            if not Path(path).exists():
                return False
        return True

    @classmethod
    def preprocess(cls, input_paths: Dict[str, Path]):
        """
        Preprocess multi-modal MRI for tumor detection using MONAI transforms.
        Uses the same preprocessing as SEGGEN5.py for consistency.
        """
        import torch
        from monai.transforms import (
            Compose, LoadImaged, EnsureChannelFirstd,
            NormalizeIntensityd, Orientationd, Spacingd
        )

        # IMPORTANT: Order must match SEGGEN5 training: [FLAIR, T1ce, T1, T2]
        modality_order = ["flair", "t1ce", "t1", "t2"]
        image_paths = [str(input_paths[m]) for m in modality_order]

        data = {"image": image_paths}

        pre_transforms = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ])

        processed = pre_transforms(data)
        return processed

    @classmethod
    def postprocess(
        cls,
        prediction: np.ndarray,
        processed_data: dict,
        reference_path: Path
    ) -> np.ndarray:
        """
        Postprocess tumor segmentation using SEGGEN5's approach.

        - Apply sigmoid and thresholds
        - Clean masks (fill holes, remove speckles)
        - Create final label map
        - Invert transforms to original space
        """
        from monai.transforms import Invertd

        # Apply sigmoid to get probabilities
        pred = prediction.sigmoid()

        # Apply thresholds (from SEGGEN5.py)
        # TC (Tumor Core) uses higher threshold to shrink necrotic core
        tc_mask = (pred[0] > 0.7).cpu().numpy()  # Tumor Core
        wt_mask = (pred[1] > 0.5).cpu().numpy()  # Whole Tumor
        et_mask = (pred[2] > 0.5).cpu().numpy()  # Enhancing Tumor

        # Clean masks
        print("Cleaning tumor masks...")
        tc_mask = clean_mask(tc_mask)
        wt_mask = clean_mask(wt_mask)
        et_mask = clean_mask(et_mask)

        # Create final segmentation with BraTS labels
        # Label 1: Necrotic (TC minus ET)
        # Label 2: Edema (WT minus TC)
        # Label 4: Enhancing (ET)
        final_seg = np.zeros(tc_mask.shape, dtype=np.uint8)
        final_seg[wt_mask] = 2  # Edema
        final_seg[tc_mask] = 1  # Necrotic core
        final_seg[et_mask] = 4  # Enhancing tumor

        # Resize to original reference shape
        ref_img = nib.load(str(reference_path))
        original_shape = ref_img.shape

        if final_seg.shape != original_shape:
            from scipy.ndimage import zoom
            zoom_factors = [o / p for o, p in zip(original_shape, final_seg.shape)]
            final_seg = zoom(final_seg, zoom_factors, order=0)

        return final_seg

    @classmethod
    def predict(
        cls,
        input_paths: Dict[str, Path],
        output_path: Path,
        reference_path: Path,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> Optional[np.ndarray]:
        """
        Run tumor segmentation on multi-modal MRI.

        Args:
            input_paths: Dict of modality -> path (t1, t1ce, t2, flair)
            output_path: Path to save tumor segmentation
            reference_path: Reference NIfTI for affine/shape
            progress_callback: Optional progress callback

        Returns:
            Tumor segmentation array or None if requirements not met
        """
        # Check if we have all required modalities
        if not cls.check_multimodal_input(input_paths):
            print("Tumor detection skipped: multi-modal input not available")
            print(f"Required modalities: {REQUIRED_MODALITIES}")
            print(f"Available: {list(input_paths.keys())}")
            return None

        if not cls.is_available():
            print("Tumor detection skipped: MONAI/PyTorch not installed")
            return None

        try:
            import torch
            from monai.inferers import sliding_window_inference

            if progress_callback:
                progress_callback(55)

            # Load model
            model = cls.load_model()
            device = cls._device

            if progress_callback:
                progress_callback(60)

            # Preprocess using MONAI transforms
            print("Preprocessing multi-modal input...")
            processed = cls.preprocess(input_paths)
            input_tensor = processed["image"].unsqueeze(0).to(device)

            if progress_callback:
                progress_callback(65)

            # Run sliding window inference (from SEGGEN5.py)
            print("Running 3D sliding window tumor inference...")
            with torch.no_grad():
                output = sliding_window_inference(
                    inputs=input_tensor,
                    roi_size=(128, 128, 128),
                    sw_batch_size=1,
                    predictor=model,
                    overlap=0.7  # High overlap for smoother results
                )

            if progress_callback:
                progress_callback(70)

            # Postprocess
            print("Postprocessing tumor segmentation...")
            tumor_seg = cls.postprocess(output[0], processed, reference_path)

            # Remap BraTS labels to our internal format (100, 101, 102)
            tumor_seg_remapped = remap_brats_labels(tumor_seg)

            # Save result using reference affine
            ref_img = nib.load(str(reference_path))
            tumor_img = nib.Nifti1Image(tumor_seg_remapped.astype(np.int16), ref_img.affine)
            nib.save(tumor_img, str(output_path))

            # Check if any tumor was found
            tumor_voxels = np.sum(tumor_seg > 0)
            if tumor_voxels > 0:
                print(f"Tumor detected: {tumor_voxels} voxels")
            else:
                print("No tumor detected")

            if progress_callback:
                progress_callback(75)

            return tumor_seg_remapped

        except Exception as e:
            raise InferenceError(f"Tumor inference failed: {e}")


def run_tumor_detection(
    input_paths: Dict[str, Path],
    output_path: Path,
    reference_path: Path,
    progress_callback: Optional[Callable[[int], None]] = None
) -> Optional[np.ndarray]:
    """Convenience function to run tumor detection."""
    return TumorInference.predict(
        input_paths,
        output_path,
        reference_path,
        progress_callback
    )
