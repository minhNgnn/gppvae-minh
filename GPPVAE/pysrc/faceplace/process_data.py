import sys
from utils import download, unzip
import sys
import glob
import scipy as sp
import scipy.linalg as la
import imageio
from PIL import Image
import numpy as np
import os
import h5py

# where have been downloaded
# Use the existing data directory, or allow override via command line
if len(sys.argv) > 1:
    data_dir = sys.argv[1]
else:
    # Default to the data/faceplace directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "../../../data/faceplace")

def main():

    # 1. download and unzip data (commented out since data is already downloaded)
    # download_data(data_dir)

    # 2. load data: reads all face images and their metadata
    RV = import_data()

    # 3. split train, validation and test: randomly splits data 80/10/10
    RV = split_data(RV)

    # 4. export: save all data to HDF5 file for fast loading during training
    out_file = os.path.join(data_dir, "data_faces.h5")
    fout = h5py.File(out_file, "w")
    # Creates datasets: Y_train, Y_val, Y_test, Did_train, Did_val, Did_test, etc.
    for key in RV.keys():
        fout.create_dataset(key, data=RV[key])
    fout.close()


def unzip_data():

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    fnames = [
        "asian.zip",
        "africanamerican.zip",
        "caucasian.zip",
        "hispanic.zip",
        "multiracial.zip",
    ]

    for fname in fnames:
        print(".. unzipping")
        unzip(os.path.join(data_dir, fname), data_dir)


def import_data(size=128):

    files = []
    # Different face orientations: 00F = frontal, 30L/R = 30 degrees left/right, etc.
    orients = ["00F", "30L", "30R", "45L", "45R", "60L", "60R", "90L", "90R"]
    for orient in orients:
        _files = glob.glob(os.path.join(data_dir, "*/*_%s.jpg" % orient))
        files = files + _files
    files = sp.sort(files)

    # Did: Dataset/Person ID (unique identifier for each person)
    Did = []
    # Rid: Rotation/View ID (the orientation/angle of the face: 00F, 30L, etc.)
    Rid = []
    # Y: Image data array (N x height x width x channels)
    Y = sp.zeros([len(files), size, size, 3], dtype=sp.uint8)

    for _i, _file in enumerate(files):
        y = imageio.imread(_file)
        # Use PIL for resizing instead of scipy.misc.imresize
        y = np.array(Image.fromarray(y).resize((size, size), Image.BILINEAR))
        Y[_i] = y
        # Parse filename: format is "PERSONID_SESSIONID_ORIENTATION.jpg"
        # Example: "BF0601_1100_00F.jpg" -> person BF0601, session 1100, frontal view
        fn = _file.split(".jpg")[0]
        fn = fn.split("/")[-1]
        did1, did2, rid = fn.split("_")
        # Combine person and session ID: "BF0601_1100"
        Did.append(did1 + "_" + did2)
        # Store orientation: "00F", "30L", etc.
        Rid.append(rid)
        
    # Convert to numpy arrays with byte string type
    Did = sp.array(Did, dtype="|S100")
    Rid = sp.array(Rid, dtype="|S100")

    # RV: Return Value dictionary containing:
    # Y = images, Did = person/session IDs, Rid = view/rotation IDs
    RV = {"Y": Y, "Did": Did, "Rid": Rid}
    return RV


def split_data(RV):

    sp.random.seed(0)
    # Split data: 80% train, 10% test, 10% validation
    n_train = int(4 * RV["Y"].shape[0] / 5.0)  
    n_test = int(1 * RV["Y"].shape[0] / 10.0)  

    # Randomly shuffle indices
    idxs = sp.random.permutation(RV["Y"].shape[0])
    idxs_train = idxs[:n_train]
    idxs_test = idxs[n_train : (n_train + n_test)]
    idxs_val = idxs[(n_train + n_test) :]  # Remaining 10% for validation

    # Create boolean masks for indexing
    # Itrain: boolean array indicating which samples are in training set
    Itrain = sp.in1d(sp.arange(RV["Y"].shape[0]), idxs_train)
    # Itest: boolean array indicating which samples are in test set
    Itest = sp.in1d(sp.arange(RV["Y"].shape[0]), idxs_test)
    # Ival: boolean array indicating which samples are in validation set
    Ival = sp.in1d(sp.arange(RV["Y"].shape[0]), idxs_val)

    out = {}
    # Split each dataset (Y, Did, Rid) into train/val/test
    for key in RV.keys():
        out["%s_train" % key] = RV[key][Itrain]
        out["%s_val" % key] = RV[key][Ival]
        out["%s_test" % key] = RV[key][Itest]

    return out


if __name__ == "__main__":

    main()
