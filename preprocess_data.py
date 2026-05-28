import numpy as np
from scipy.ndimage import gaussian_filter
import nibabel as nib

# Input: 'img' is a 4D image array,
#        'mask' is a corresponding foreground mask (values 0 and 1 on 3D array),
#        'window_size' is the size of the neighbourhood extracted for each foreground voxel
# Returns two things: 1) a 2D array with flattened neigbourhoods for each foreground voxel (shape: #foreground,
# #neigbourhood * time frames) and 2) a 2D array containing the foreground indices (shape: #foreground, 3(=x,y,z))
def extract_tacs(img, mask, window_size=(5, 5, 3)):

    # Formulate foreground indices to 2D array
    inds = np.where(mask > 0.5)
    indices = np.transpose(np.asarray(inds))

    # Initialise TACs array and neighbourhood radius
    tacs = np.full((len(inds[0]), np.prod(window_size)*img.shape[3]), np.nan)
    window = ((np.asarray(window_size) - 1) / 2).astype(int)

    # Pad the image so that borders don't cause problems
    buffer_size = ((window[0], window[0]), (window[1], window[1]), (window[2], window[2]), (0, 0))
    img_padded = np.pad(img, buffer_size, mode='edge')

    # Collect neighbourhood TACs
    for index in range(len(inds[0])):
        ind = indices[index]
        x_range = range(ind[0], ind[0] + window[0] + window[0] + 1, 1)  
        y_range = range(ind[1], ind[1] + window[1] + window[1] + 1, 1)  
        z_range = range(ind[2], ind[2] + window[2] + window[2] + 1, 1)
        flat_tac = img_padded[np.ix_(x_range, y_range, z_range, range(img_padded.shape[3]))].flatten()
        tacs[index, :] = flat_tac

    return tacs, indices

"""
Preprocessing of an example dataset (based on our test data)
"""
for img_id in ["list", "of", "image", "ids"]:

    # Read in PET and foreground mask
    pet = nib.load("path/to/pet/PET_" + img_id + ".nii").get_fdata()
    mask = nib.load("path/to/mask/Mask_" + img_id + ".nii").get_fdata()

    # Denoise PET
    pet = gaussian_filter(pet, sigma=0.75)

    # Harmonise time frames (this needs to me manually edited for each dataset)
    dims = pet.shape
    preprocessed = np.zeros((dims[0], dims[1], dims[2], 18))
    preprocessed[:, :, :, 3:9] = pet[:, :, :, 0:6]
    preprocessed[:, :, :, 9] = np.mean(pet[:, :, :, 6:8], axis=3)
    preprocessed[:, :, :, 10] = np.mean(pet[:, :, :, 8:10], axis=3)
    preprocessed[:, :, :, 11] = np.mean(pet[:, :, :, 9:11], axis=3)
    preprocessed[:, :, :, 12] = np.mean(pet[:, :, :, 10:12], axis=3)
    preprocessed[:, :, :, 13] = np.average(pet[:, :, :, 11:14], axis=3, weights=[0.5, 1, 1])
    preprocessed[:, :, :, 14] = np.average(pet[:, :, :, 14:16], axis=3, weights=[1, 0.25])
    preprocessed[:, :, :, 15] = np.average(pet[:, :, :, 15:17], axis=3, weights=[3, 1])
    preprocessed[:, :, :, 16] = np.average(pet[:, :, :, 16:18], axis=3, weights=[3, 1])
    preprocessed[:, :, :, 17] = np.average(pet[:, :, :, 16:18], axis=3, weights=[0.75, 0.5])

    # Reduce the resolution to match other datasets (dim2 is close enough (2.79 vs 3.27), I won't adjust that)
    preprocessed = preprocessed[::2, ::2, :, :]
    mask = mask[::2, ::2, :]
    mask = mask.astype(np.bool)

    # Extract voxel neighbourhoods
    tacs, inds = extract_tacs(preprocessed, mask)  # inds are needed when the predicted values are converted back to 3D array

    # Logistic scaling
    tacs = (tacs - np.nanmean(tacs_foreground)) / np.nanstd(tacs_foreground)
    tacs = 1 / (1 + np.exp(-0.8 * tacs))
    
    # Save preprocessed data
    np.save("path/to/input/TACs_" + img_id + ".npy", tacs)
    np.save("path/to/input/Inds_" + img_id + ".npy", inds)
