import os
import nibabel as nib
import numpy as np

base_path = r'C:\Users\prana\Downloads\Totalsegmentator_dataset_small_v201'

def find_valid_subjects():
    for subject in os.listdir(base_path):
        brain_path = os.path.join(base_path, subject, 'segmentations', 'brain.nii.gz')
        if os.path.exists(brain_path):
            data = nib.load(brain_path).get_fdata()
            if np.max(data) > 0:
                print(f"✅ FOUND BRAIN DATA in subject: {subject}")
                return subject
    print("❌ No subjects in this small subset contain brain data.")
    return None

valid_subject = find_valid_subjects()