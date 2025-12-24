import sys
from utils import download, unzip
import sys
import glob
import numpy as np
import imageio
from PIL import Image
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
    print("=" * 80)
    print("IMPORTING DATA")
    print("=" * 80)
    RV = import_data()
    
    print("\n✅ Data import complete!")
    print(f"   Total samples: {RV['Y'].shape[0]}")
    print(f"   Image shape: {RV['Y'].shape[1:]}")
    print(f"   Unique people: {len(np.unique(RV['Did']))}")
    print(f"   Unique views: {len(np.unique(RV['Rid']))}")
    
    # Show first 27 samples to verify angular ordering (3 people x 9 views)
    print("\n" + "=" * 80)
    print("FIRST 27 SAMPLES (verify angular ordering):")
    print("=" * 80)
    print(f"{'Index':<8} {'Person ID':<20} {'View':<8} {'Expected Angle':<25}")
    print("-" * 80)
    
    angle_map = {
        b"90L": "-90° (left profile)",
        b"60L": "-60° (left)",
        b"45L": "-45° (left)",
        b"30L": "-30° (left)",
        b"00F": "  0° (frontal)",
        b"30R": "+30° (right)",
        b"45R": "+45° (right)",
        b"60R": "+60° (right)",
        b"90R": "+90° (right profile)",
    }
    
    for i in range(min(27, len(RV['Did']))):
        person_id = RV['Did'][i].decode('utf-8')
        view_label = RV['Rid'][i]
        angle_desc = angle_map.get(view_label, "unknown")
        print(f"{i:<8} {person_id:<20} {view_label.decode('utf-8'):<8} {angle_desc:<25}")
    
    # Show view distribution
    print("\n" + "=" * 80)
    print("VIEW DISTRIBUTION (should all be equal):")
    print("=" * 80)
    unique_views, counts = np.unique(RV['Rid'], return_counts=True)
    print(f"{'View':<8} {'Angle':<25} {'Count':<10}")
    print("-" * 80)
    for view, count in zip(unique_views, counts):
        angle_desc = angle_map.get(view, "unknown")
        print(f"{view.decode('utf-8'):<8} {angle_desc:<25} {count:<10}")
    
    # Verify each person has all 9 views in angular order
    print("\n" + "=" * 80)
    print("CHECKING DATA COMPLETENESS (first 3 people):")
    print("=" * 80)
    people = np.unique(RV['Did'])
    for person in people[:3]:
        mask = RV['Did'] == person
        person_views = RV['Rid'][mask]
        print(f"\nPerson {person.decode('utf-8')}:")
        print(f"  Views: {[v.decode('utf-8') for v in person_views]}")
        print(f"  Count: {len(person_views)}/9")

    # 3. split train, validation and test: randomly splits data 80/10/10
    print("\n" + "=" * 80)
    print("SPLITTING DATA (80% train, 10% val, 10% test)")
    print("=" * 80)
    RV = split_data(RV)
    print("✅ Split complete!")

    # 4. export: save all data to HDF5 file for fast loading during training
    print("\n" + "=" * 80)
    print("SAVING TO HDF5")
    print("=" * 80)
    out_file = os.path.join(data_dir, "data_faces.h5")
    print(f"Output file: {out_file}")
    fout = h5py.File(out_file, "w")
    # Creates datasets: Y_train, Y_val, Y_test, Did_train, Did_val, Did_test, etc.
    for key in RV.keys():
        fout.create_dataset(key, data=RV[key])
        print(f"  Created dataset: {key} with shape {RV[key].shape}")
    fout.close()
    print("✅ HDF5 file saved successfully!")


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
    # ORDER BY ANGLE (left to right): -90° to +90°
    # This ensures view indices follow geometric progression for kernel methods
    orients = ["90L", "60L", "45L", "30L", "00F", "30R", "45R", "60R", "90R"]
    #          -90°   -60°   -45°   -30°    0°    +30°   +45°   +60°   +90°
    #          idx:0  idx:1  idx:2  idx:3  idx:4  idx:5  idx:6  idx:7  idx:8
    for orient in orients:
        _files = glob.glob(os.path.join(data_dir, "*/*_%s.jpg" % orient))
        files = files + _files
    # DON'T sort alphabetically - preserve angular order!
    # files = np.sort(files)  # ← Removed to keep angular ordering

    # Did: Dataset/Person ID (unique identifier for each person)
    Did = []
    # Rid: Rotation/View ID (the orientation/angle of the face: 00F, 30L, etc.)
    Rid = []
    # Y: Image data array (N x height x width x channels)
    Y = np.zeros([len(files), size, size, 3], dtype=np.uint8)

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
    Did = np.array(Did, dtype="|S100")
    Rid = np.array(Rid, dtype="|S100")

    # RV: Return Value dictionary containing:
    # Y = images, Did = person/session IDs, Rid = view/rotation IDs
    RV = {"Y": Y, "Did": Did, "Rid": Rid}
    return RV


def split_data(RV):

    np.random.seed(0)
    # Split data: 80% train, 10% test, 10% validation
    n_train = int(4 * RV["Y"].shape[0] / 5.0)  
    n_test = int(1 * RV["Y"].shape[0] / 10.0)  

    # Randomly shuffle indices
    idxs = np.random.permutation(RV["Y"].shape[0])
    idxs_train = idxs[:n_train]
    idxs_test = idxs[n_train : (n_train + n_test)]
    idxs_val = idxs[(n_train + n_test) :]  # Remaining 10% for validation

    # Create boolean masks for indexing
    # Itrain: boolean array indicating which samples are in training set
    Itrain = np.in1d(np.arange(RV["Y"].shape[0]), idxs_train)
    # Itest: boolean array indicating which samples are in test set
    Itest = np.in1d(np.arange(RV["Y"].shape[0]), idxs_test)
    # Ival: boolean array indicating which samples are in validation set
    Ival = np.in1d(np.arange(RV["Y"].shape[0]), idxs_val)

    out = {}
    # Split each dataset (Y, Did, Rid) into train/val/test
    for key in RV.keys():
        out["%s_train" % key] = RV[key][Itrain]
        out["%s_val" % key] = RV[key][Ival]
        out["%s_test" % key] = RV[key][Itest]

    return out


if __name__ == "__main__":

    main()
