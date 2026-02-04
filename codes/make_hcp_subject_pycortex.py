"""
-----------------------------------------------------------------------------------------
make_hcp_subject_pycortex.py
-----------------------------------------------------------------------------------------
Goal of the script:
Import HCP 32k and 59k subject in pycortex database
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
>> cd ~/projects/HCP_pycortex_subjects/codes
2. run python command
python make_hcp_subject_pycortex.py [main directory] [pycortex directory]
-----------------------------------------------------------------------------------------
Executions:
cd ~/projects/HCP_pycortex_subjects/codes
python make_hcp_subject_pycortex.py /scratch/ulascombes/data/hcp_data /home/ulascombes/projects/HCP_pycortex_subjects/data
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
import cortex
import importlib
import numpy as np
import nibabel as nb

# Personal imports 
sys.path.append("{}/utils".format(os.getcwd()))
from pycortex_utils import set_pycortex_config_file, setup_pycortex_dirs

# Inputs
hcp_data_dir = sys.argv[1]
pycortex_dir = sys.argv[2]

# Set pycortex db and colormaps
cortex_dir = "{}/cortex".format(pycortex_dir)

# Setup pycortex directories
setup_pycortex_dirs(cortex_dir)

formats = ['32k', '59k']
HCP_subject = '100610'

for n, format_ in enumerate(formats):
    subject = 'sub-hcp_{}'.format(format_)
    
    # Set pycortex db and colormaps
    set_pycortex_config_file(cortex_dir)
    importlib.reload(cortex)
  
    if '32k' in subject : 
        res = '_'
    elif '59k' in subject : 
        res = '_1.6mm_'
    
    # Make pycortex subject
    cortex.db.make_subj(subject)

    # Copy relevant data
    print('coping data ...')
    hcp_T1w_dir = '{}/{}/T1w'.format(hcp_data_dir, HCP_subject)
    hcp_MNINonLinear_dir = '{}/{}/MNINonLinear'.format(hcp_data_dir, HCP_subject)
    
    hcp_template_dir = '{}/fsaverage_LR{}'.format(hcp_MNINonLinear_dir, format_)
    
    
    pycortex_surf_dir = '{}/db/{}/surfaces'.format(cortex_dir, subject)
    pycortex_anat_dir = '{}/db/{}/anatomicals'.format(cortex_dir, subject)
    pycortex_surf_inf_dir = '{}/db/{}/surface-info'.format(cortex_dir, subject)
    
    # wm (white matter)
    shutil.copy('{}/{}.L.white{}MSMAll.{}_fs_LR.surf.gii'.format(
        hcp_template_dir, HCP_subject, res, format_),
        '{}/wm_lh.gii'.format(pycortex_surf_dir))
    shutil.copy('{}/{}.R.white{}MSMAll.{}_fs_LR.surf.gii'.format(hcp_template_dir, HCP_subject, res, format_), 
                '{}/wm_rh.gii'.format(pycortex_surf_dir))
    print('wm is done')
    
    # pial
    shutil.copy('{}/{}.L.pial{}MSMAll.{}_fs_LR.surf.gii'.format(
        hcp_template_dir, HCP_subject, res, format_),
        '{}/pia_lh.gii'.format(pycortex_surf_dir))
    shutil.copy('{}/{}.R.pial{}MSMAll.{}_fs_LR.surf.gii'.format(
        hcp_template_dir, HCP_subject, res, format_),
        '{}/pia_rh.gii'.format(pycortex_surf_dir))
    print('pial is done')
    
    # inflated
    shutil.copy('{}/{}.L.inflated{}MSMAll.{}_fs_LR.surf.gii'.format(
        hcp_template_dir, HCP_subject, res, format_),
        '{}/inflated_lh.gii'.format(pycortex_surf_dir))
    shutil.copy('{}/{}.R.inflated{}MSMAll.{}_fs_LR.surf.gii'.format(
        hcp_template_dir, HCP_subject, res, format_),
        '{}/inflated_rh.gii'.format(pycortex_surf_dir))
    print('inflated is done')
    
    # very inflated
    shutil.copy('{}/{}.L.very_inflated{}MSMAll.{}_fs_LR.surf.gii'.format(
        hcp_template_dir, HCP_subject, res, format_),
        '{}/very_inflated_lh.gii'.format(pycortex_surf_dir))
    shutil.copy('{}/{}.R.very_inflated{}MSMAll.{}_fs_LR.surf.gii'.format(
        hcp_template_dir, HCP_subject, res, format_),
        '{}/very_inflated_rh.gii'.format(pycortex_surf_dir))
    print('very inflated is done')
    
    # flat
    shutil.copy('{}/{}.L.flat.{}_fs_LR.surf.gii'.format(
        hcp_template_dir, HCP_subject, format_),
        '{}/flat_lh.gii'.format(pycortex_surf_dir))
    shutil.copy('{}/{}.R.flat.{}_fs_LR.surf.gii'.format(
        hcp_template_dir, HCP_subject, format_),
        '{}/flat_rh.gii'.format(pycortex_surf_dir))
    print('flat is done')
    
    # raw T1w
    shutil.copy('{}/T1w_acpc_dc_restore.nii.gz'.format(hcp_T1w_dir),
                '{}/raw.nii.gz'.format(pycortex_anat_dir))
    print('raw T1w inflated is done')
    
    # aseg segmentation
    shutil.copy('{}/aparc+aseg.nii.gz'.format(hcp_T1w_dir),
                '{}/aseg.nii.gz'.format(pycortex_anat_dir))
    print('aseg inflated is done')
    
    # white matter volume
    shutil.copy('{}/wmparc.nii.gz'.format(hcp_MNINonLinear_dir),
                '{}/raw_wm.nii.gz'.format(pycortex_anat_dir))
    print('white matter volume inflated is done')
    
    
    # brainmask
    shutil.copy('{}/brainmask_fs.nii.gz'.format(hcp_T1w_dir), 
                '{}/brainmask.nii.gz'.format(pycortex_anat_dir))
    print('brainmask volume inflated is done')
    
        
    # Sulcaldepth
    # Load brain mask 
    mask_full_L_fn = '{}/{}.L.atlasroi.{}_fs_LR.shape.gii'.format(hcp_template_dir, HCP_subject, format_)
    mask_full_L_img = nb.load(mask_full_L_fn)
    mask_full_L_data = mask_full_L_img.darrays[0].data
    n_vert_full_L = mask_full_L_data.shape[0]
    
    mask_full_R_fn = '{}/{}.R.atlasroi.{}_fs_LR.shape.gii'.format(hcp_template_dir, HCP_subject, format_)
    mask_full_R_img = nb.load(mask_full_R_fn)
    mask_full_R_data = mask_full_R_img.darrays[0].data
    n_vert_full_R = mask_full_R_data.shape[0]
    
    mask_full_brain_data = np.concatenate([mask_full_L_data, mask_full_R_data])
    
    # Save cortex_mask
    mask_dict = {'left' : mask_full_L_data.astype(int).astype(bool), 
                 'right' : mask_full_R_data.astype(int).astype(bool)}
    
    print('Save cortex mask')
    np.savez('{}/cortex_mask.npz'.format(pycortex_surf_inf_dir), **mask_dict)
    
    # Load sulcaldepth
    sulc_cortex_brain_fn = "{}/{}.sulc{}MSMAll.{}_fs_LR.dscalar.nii".format(hcp_template_dir, HCP_subject, res, format_)
    sulc_cortex_brain_img = nb.load(sulc_cortex_brain_fn)
    sulc_cortex_brain_data = sulc_cortex_brain_img.get_fdata().squeeze()
    
    # make 64k sulcaldepth
    sulc_full_brain_data = np.zeros_like(mask_full_brain_data)
    sulc_full_brain_data[mask_full_brain_data == 1] = sulc_cortex_brain_data
    
    sulc_dict = {'left' : sulc_full_brain_data[:n_vert_full_L], 
                 'right' : sulc_full_brain_data[n_vert_full_L:]}
    
    print('Save sulcaldepth')
    np.savez('{}/sulcaldepth.npz'.format(pycortex_surf_inf_dir), **sulc_dict)
    
    # Make overlay
    surfs = [cortex.polyutils.Surface(*d) for d in cortex.db.get_surf(subject, "fiducial")]
    num_verts = surfs[0].pts.shape[0] + surfs[1].pts.shape[0]
    rand_data = np.random.randn(num_verts)
    vertex_data = cortex.Vertex(rand_data, subject)
    ds = cortex.Dataset(rand=vertex_data)
    
    
    temp_dir = "{}/temp_data/{}_rand_ds/".format(pycortex_dir, subject)
    cortex.webgl.make_static(outpath=temp_dir, data=ds, types=('inflated','very_inflated'))

    
    
    
    