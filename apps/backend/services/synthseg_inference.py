"""Brain segmentation using ANTsPyNet (replaces SynthSeg for Python 3.12+ compatibility)."""
from pathlib import Path
from typing import Callable, Optional

import nibabel as nib
import numpy as np

from services.model_manager import InferenceError


def run_synthseg(
    input_path: Path,
    output_path: Path,
    flair_path: Optional[Path] = None,
    t2_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[int], None]] = None
) -> np.ndarray:
    """
    Run brain segmentation using ANTsPyNet's deep_atropos.

    This provides tissue segmentation (CSF, gray matter, white matter, etc.)
    similar to SynthSeg but compatible with modern Python versions.

    If flair_path is provided, uses FLAIR for brain extraction (cleaner mask).
    If t2_path is provided, uses T1+T2 multi-modal mode for better subcortical
    structure segmentation (deep gray matter, brainstem, cerebellum).

    Note: No preprocessing is done - deep_atropos handles N4 bias correction internally.

    Args:
        input_path: Path to T1 NIfTI file (NOT T1ce - model trained on regular T1)
        output_path: Path to save segmentation NIfTI
        flair_path: Optional FLAIR image for better brain extraction
        t2_path: Optional T2 image for multi-modal segmentation
        progress_callback: Optional callback for progress updates

    Returns:
        Segmentation as numpy array with tissue labels
    """
    try:
        if progress_callback:
            progress_callback(5)

        print("Loading ANTsPyNet for brain segmentation...")

        import ants
        import antspynet

        if progress_callback:
            progress_callback(10)

        # Load the T1 image
        print(f"Loading T1 image: {input_path}")
        t1_img = ants.image_read(str(input_path))

        if progress_callback:
            progress_callback(20)

        # Run brain extraction
        # Use FLAIR if available (better brain-skull boundary detection)
        if flair_path and flair_path.exists():
            print("Using FLAIR for brain extraction...")
            flair_img = ants.image_read(str(flair_path))

            brain_extraction = antspynet.brain_extraction(
                flair_img,
                modality="flair",
                verbose=True
            )
            brain_mask = brain_extraction["brain_mask"]

            # Resample mask to T1 space if needed
            if not ants.image_physical_space_consistency(brain_mask, t1_img):
                print("Resampling FLAIR brain mask to T1 space...")
                brain_mask = ants.resample_image_to_target(
                    brain_mask, t1_img, interp_type="nearestNeighbor"
                )
        else:
            print("Using T1-based brain extraction...")
            brain_extraction = antspynet.brain_extraction(
                t1_img,
                modality="t1",
                verbose=True
            )
            brain_mask = brain_extraction["brain_mask"]

        if progress_callback:
            progress_callback(40)

        # Apply mask to get brain-only T1 image
        t1_brain = t1_img * brain_mask

        if progress_callback:
            progress_callback(45)

        # Run deep_atropos for tissue segmentation
        # This segments into: CSF, gray matter, white matter, deep gray matter, brainstem, cerebellum
        if t2_path and t2_path.exists():
            # Multi-modal mode: T1 + T2 (much better for subcortical structures)
            print("Running multi-modal deep_atropos (T1 + T2)...")
            t2_img = ants.image_read(str(t2_path))

            # Resample T2 to T1 space if needed
            if not ants.image_physical_space_consistency(t2_img, t1_img):
                print("Resampling T2 to T1 space...")
                t2_img = ants.resample_image_to_target(
                    t2_img, t1_img, interp_type="linear"
                )

            # Apply brain mask to T2
            t2_brain = t2_img * brain_mask

            # deep_atropos multi-modal: expects [T1, T2, FA] - pass None for FA
            atropos_result = antspynet.deep_atropos(
                [t1_brain, t2_brain, None],
                do_preprocessing=True,
                verbose=True
            )
        else:
            # Single-modal mode: T1 only
            print("Running single-modal deep_atropos (T1 only)...")
            atropos_result = antspynet.deep_atropos(
                t1_brain,
                do_preprocessing=True,
                verbose=True
            )

        if progress_callback:
            progress_callback(75)

        # Get the segmentation image
        segmentation_img = atropos_result["segmentation_image"]

        # Convert to numpy and remap labels to match our expected format
        # deep_atropos labels: 0=background, 1=CSF, 2=gray matter, 3=white matter,
        #                      4=deep gray matter, 5=brainstem, 6=cerebellum
        segmentation = segmentation_img.numpy().astype(np.int32)

        # Remap to FreeSurfer-like labels for compatibility with our label system
        label_remap = {
            0: 0,    # Background
            1: 24,   # CSF
            2: 3,    # Gray matter -> left cerebral cortex
            3: 2,    # White matter -> left cerebral white matter
            4: 10,   # Deep gray matter -> left thalamus (subcortical)
            5: 16,   # Brainstem
            6: 8,    # Cerebellum -> left cerebellum cortex
        }

        remapped = np.zeros_like(segmentation)
        for old_label, new_label in label_remap.items():
            remapped[segmentation == old_label] = new_label

        if progress_callback:
            progress_callback(85)

        # Save the segmentation
        print(f"Saving segmentation to: {output_path}")

        # Get the original affine from nibabel for consistency
        orig_nib = nib.load(str(input_path))
        seg_nib = nib.Nifti1Image(remapped.astype(np.int16), orig_nib.affine)
        nib.save(seg_nib, str(output_path))

        # Report results
        unique_labels = np.unique(remapped[remapped > 0])
        print(f"Segmentation complete. Found {len(unique_labels)} tissue types: {unique_labels.tolist()}")

        if progress_callback:
            progress_callback(90)

        return remapped

    except ImportError as e:
        raise InferenceError(
            f"ANTsPyNet not installed. Run: pip install antspyx antspynet\nError: {e}"
        )
    except Exception as e:
        raise InferenceError(f"Brain segmentation failed: {e}")
