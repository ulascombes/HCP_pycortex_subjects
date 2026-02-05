"""
-----------------------------------------------------------------------------------------
project_mmp_pycortex.py
-----------------------------------------------------------------------------------------
Goal of the script:
Import subject in pycortex database
-----------------------------------------------------------------------------------------
Input(s):
sys.argv[1]: hcp subject dir
sys.argv[2]: pycortex dir
-----------------------------------------------------------------------------------------
Output(s):
# Preprocessed and averaged timeseries files
-----------------------------------------------------------------------------------------
To run:
1. cd to function
>> cd ~/disks/meso_H/projects/HCP_pycortex_subjects/codes
2. run python command
python project_mmp_pycortex.py [main directory]
-----------------------------------------------------------------------------------------
Executions:
cd ~/disks/meso_H/projects/HCP_pycortex_subjects/codes
python project_mmp_pycortex.py /Users/uriel/disks/meso_H/projects/HCP_pycortex_subjects/data
-----------------------------------------------------------------------------------------
Written by Uriel Lascombes (uriel.lascombes@laposte.net)
-----------------------------------------------------------------------------------------
"""

# stop warnings
import warnings
warnings.filterwarnings("ignore")

# Debug
import ipdb
deb = ipdb.set_trace

# Import
import os
import sys
import shutil
import numpy as np
import pandas as pd
import nibabel as nb

# Personal imports
sys.path.append("/Users/uriel/disks/meso_H/projects/HCP_pycortex_subjects/codes/utils")
from pycortex_rois_utils import *
from pycortex_utils import draw_cortex, set_pycortex_config_file

# Imputs
main_dir = sys.argv[1]
formats = ['32k', '59k']

# Set pycortex db and colormaps
cortex_dir = "{}/cortex".format(main_dir)
set_pycortex_config_file(cortex_dir)

for n, format_ in enumerate(formats):
    subject = 'sub-hcp{}'.format(format_)
    
    # Load mmp dlabel
    mmp_dlabel_fn = '{}/atlas/HCP_MMP1.Glasser.{}_fs_LR.dlabel.nii'.format(main_dir, format_)
    mmp_label_partial_data = nb.load(mmp_dlabel_fn).get_fdata().squeeze()
    
    # Load mask 
    mask_32k_fn = '{}/cortex/db/{}/surface-info/cortex_mask.npz'.format(main_dir, subject)
    mask_32k_npz = np.load(mask_32k_fn)
    mask_brain = np.concatenate([mask_32k_npz['left'], mask_32k_npz['right']]).astype(int)
    
    n_vert_full_L = mask_32k_npz['left'].shape[0]
    n_vert_full_R = mask_32k_npz['right'].shape[0]
    
    if format_ == '32k':
        # Adding medial wall vetices 
        mmp_label_full_data = np.zeros_like(mask_brain)
        mmp_label_full_data[mask_brain == 1] = mmp_label_partial_data
        
    elif format_ == '59k':
        mmp_label_full_data = mmp_label_partial_data
        
    # Transform label to be between 0 and 180
    mmp_label_full_data = ((mmp_label_full_data - 1) % 180) + 1
    
    # split hemisperes
    mmp_label_full_L_data = mmp_label_full_data[:n_vert_full_L]
    mmp_label_full_R_data = mmp_label_full_data[n_vert_full_L:]
    
    # Load mmp rois names 
    mmp_label_fn = '{}/atlas/mmp_rois_numbers.tsv'.format(main_dir)
    mmp_label_df = pd.read_table(mmp_label_fn, sep="\t")
    
    # Make mmp npz
    roi_dict_brain = {}
    roi_dict_left = {}
    roi_dict_right = {}
    
    for _, row in mmp_label_df.iterrows():
        roi_name = row["roi_name"]
        roi_num = row["roi_num"]
    
        roi_dict_brain[roi_name] = (mmp_label_full_data == roi_num)
        roi_dict_left[roi_name]  = (mmp_label_full_L_data == roi_num)
        roi_dict_right[roi_name] = (mmp_label_full_R_data == roi_num)
    
    mmp_label_npz_dir = '{}/cortex/db/{}/rois'.format(main_dir, subject)
    os.makedirs(mmp_label_npz_dir, exist_ok=True)
    
    np.savez('{}/{}_rois-mmp.npz'.format(mmp_label_npz_dir, format_), **roi_dict_brain)
    np.savez('{}/{}_hemi-L_rois-mmp.npz'.format(mmp_label_npz_dir, format_), **roi_dict_left)
    np.savez('{}/{}_hemi-R_rois-mmp.npz'.format(mmp_label_npz_dir, format_), **roi_dict_right)
    
    # Add the mmp on overlay 
    colormap_name = 'HCP_MMP1'
    alpha = np.ones(mmp_label_full_data.shape)
    roi_name = 'glasser-mmp'
    param_rois = {'subject': subject,
                  'data': mmp_label_full_data, 
                  'cmap': colormap_name,
                  'alpha': alpha,
                  'cbar': 'discrete', 
                  'vmin': 0,
                  'vmax': 255,
                  'cmap_steps': 255,
                  'cortex_type': 'VertexRGB',
                  'description': 'Gaussian pRF ROIs',
                  'curv_brightness': 1, 
                  'curv_contrast': 0.25,
                  'add_roi': True,
                  'with_labels': True,
                  'roi_name': roi_name}
    
    # Plot
    volume_roi = draw_cortex(**param_rois)
    
    # project roi borders on overlay 
    overlay_mmp_fn = '{}/cortex/db/{}/overlays.svg'.format(main_dir, subject)
    overlay_group_mmp_fn = '{}/cortex/db/{}/overlays_rois-group-mmp.svg'.format(main_dir, subject)
    shutil.copy(overlay_mmp_fn, overlay_group_mmp_fn)
    
    rp = ROIpack(subject, '{}/{}_rois-mmp.npz'.format(mmp_label_npz_dir, format_))
    rp.to_svg(filename=overlay_group_mmp_fn)
    
