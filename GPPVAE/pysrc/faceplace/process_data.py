import sys
from utils import download, unzip
import sys
import glob
import numpy as np
import imageio
from PIL import Image
import os
import h5py

# Configuration: ethnicity filtering
# Options: 
#   None = use all ethnicities (imbalanced: 45% caucasian, 14% african-american)
#   ['african-american'] = only black faces
#   ['caucasian'] = only white faces
#   ['asian', 'caucasian', 'african-american', 'hispanic', 'multiracial'] = all, but can balance
ETHNICITY_FILTER = None  # Set to list of folder names or None for all

# Whether to balance classes by undersampling (take min count from each ethnicity)
# If True: will undersample to 432 images per ethnicity (smallest class size)
# Result: 5 ethnicities × 432 = 2,160 total images (balanced 20% each)
BALANCE_CLASSES = False  # Set to True to balance ethnicities

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

    # ORDER BY ANGLE (left to right): -90° to +90°
    # This ensures view indices follow geometric progression for kernel methods
    orients = ["90L", "60L", "45L", "30L", "00F", "30R", "45R", "60R", "90R"]
    #          -90°   -60°   -45°   -30°    0°    +30°   +45°   +60°   +90°
    #          idx:0  idx:1  idx:2  idx:3  idx:4  idx:5  idx:6  idx:7  idx:8
    
    # Step 1: Collect all files organized by (ethnicity → person → view)
    # Key: (ethnicity, person_id), Value: dict of {orient: filepath}
    person_files = {}
    
    for orient in orients:
        if ETHNICITY_FILTER is None:
            # Load all ethnicities
            _files = glob.glob(os.path.join(data_dir, "*/*_%s.jpg" % orient))
        else:
            # Load only specified ethnicities
            _files = []
            for eth in ETHNICITY_FILTER:
                eth_files = glob.glob(os.path.join(data_dir, eth + "/*_%s.jpg" % orient))
                _files.extend(eth_files)
        
        # Group by (ethnicity, person)
        for f in _files:
            ethnicity = f.split("/")[-2]  # Extract folder name (ethnicity)
            filename = f.split("/")[-1]
            # Extract person ID: "BF0601_1100_00F.jpg" → "BF0601_1100"
            person_id = "_".join(filename.split("_")[:2])
            
            key = (ethnicity, person_id)
            if key not in person_files:
                person_files[key] = {}
            person_files[key][orient] = f
    
    # Step 2: Filter to only keep people with ALL 9 views
    complete_people = {key: files for key, files in person_files.items() 
                      if len(files) == 9}  # Must have all 9 orientations
    
    incomplete_count = len(person_files) - len(complete_people)
    if incomplete_count > 0:
        print(f"\n⚠️  Filtered out {incomplete_count} people with incomplete views")
    
    # Step 3: Organize by ethnicity for potential balancing
    files_by_ethnicity = {}
    for (ethnicity, person_id), view_dict in complete_people.items():
        if ethnicity not in files_by_ethnicity:
            files_by_ethnicity[ethnicity] = []
        
        # Add this person's 9 views IN ANGULAR ORDER
        person_views = [view_dict[orient] for orient in orients]
        files_by_ethnicity[ethnicity].extend(person_views)
    
    # Step 4: Balance classes if requested
    if BALANCE_CLASSES and len(files_by_ethnicity) > 1:
        # Count people per ethnicity (divide by 9 since each person has 9 views)
        people_per_eth = {eth: len(files_by_ethnicity[eth]) // 9 
                         for eth in files_by_ethnicity}
        min_people = min(people_per_eth.values())
        
        print(f"\n⚖️  BALANCING CLASSES: undersampling to {min_people} people per ethnicity")
        
        balanced_files = {}
        np.random.seed(42)  # Reproducible sampling
        for eth in files_by_ethnicity:
            # Sample complete people (9 views each)
            n_people = len(files_by_ethnicity[eth]) // 9
            people_indices = np.arange(n_people)
            sampled_people = np.random.choice(people_indices, min_people, replace=False)
            
            # Extract selected people (each person = 9 consecutive files)
            sampled_files = []
            for person_idx in sorted(sampled_people):
                start = person_idx * 9
                sampled_files.extend(files_by_ethnicity[eth][start:start+9])
            
            balanced_files[eth] = sampled_files
        files_by_ethnicity = balanced_files
    
    # Step 5: Flatten to single list PRESERVING PERSON-VIEW STRUCTURE
    # IMPORTANT: Do NOT interleave by ethnicity!
    # GP-VAE requires all 9 views of each person to be together
    # The angular ordering (90L → 60L → ... → 90R) must be preserved per person
    files = []
    for eth in sorted(files_by_ethnicity.keys()):
        files.extend(files_by_ethnicity[eth])
    
    print(f"\n📋 STRUCTURE: Files organized by (ethnicity → person → views)")
    print(f"   Each person has 9 consecutive views in angular order")
    
    # Print distribution
    print(f"\n📊 ETHNICITY DISTRIBUTION:")
    print(f"{'Ethnicity':<20} {'Count':<10} {'Percentage':<12}")
    print("-" * 50)
    ethnicity_counts = {eth: len(files_by_ethnicity[eth]) for eth in files_by_ethnicity}
    total = sum(ethnicity_counts.values())
    for eth in sorted(ethnicity_counts.keys()):
        count = ethnicity_counts[eth]
        pct = 100 * count / total
        print(f"{eth:<20} {count:<10} {pct:>5.1f}%")
    print(f"{'TOTAL':<20} {total:<10} 100.0%")
    
    if ETHNICITY_FILTER is not None:
        print(f"\n🔍 FILTER APPLIED: {ETHNICITY_FILTER}")
    if BALANCE_CLASSES:
        print(f"⚖️  BALANCED: Each ethnicity has equal representation")

    print(f"\n🔍 VERIFICATION: First 15 files to process:")
    print(f"{'Index':<6} {'Ethnicity':<20} {'Filename':<40}")
    print("-" * 80)
    for i, f in enumerate(files[:15]):
        ethnicity = f.split("/")[-2]
        filename = f.split("/")[-1]
        print(f"{i:<6} {ethnicity:<20} {filename:<40}")

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
    
    # VERIFICATION: Check that parsed Rid matches actual file orientation
    print(f"\n🔍 VERIFICATION: Did/Rid extraction (first 15 samples):")
    print(f"{'Index':<6} {'Filename':<50} {'Extracted Did':<15} {'Extracted Rid':<12} {'Match?':<8}")
    print("-" * 100)
    for i in range(min(15, len(files))):
        filename = files[i].split("/")[-1]
        extracted_did = Did[i]
        extracted_rid = Rid[i]
        # Verify: does the filename contain the extracted rid?
        match = "✅" if extracted_rid in filename else "❌"
        print(f"{i:<6} {filename:<50} {extracted_did:<15} {extracted_rid:<12} {match:<8}")

    # Convert to numpy arrays with byte string type
    Did = np.array(Did, dtype="|S100")
    Rid = np.array(Rid, dtype="|S100")

    # RV: Return Value dictionary containing:
    # Y = images, Did = person/session IDs, Rid = view/rotation IDs
    RV = {"Y": Y, "Did": Did, "Rid": Rid}
    return RV


def split_data(RV):

    # CRITICAL: Split by PERSON, not by individual samples
    # Each person has 9 consecutive views in the dataset
    # We need to keep all 9 views together in the same split
    
    # seed for splitting data
    np.random.seed(0)
    
    # Get unique people and count them
    unique_people = np.unique(RV["Did"])
    n_people = len(unique_people)
    
    print(f"\n📊 Split strategy: Person-level (not sample-level)")
    print(f"   Total people: {n_people}")
    print(f"   Samples per person: 9 views")
    print(f"   Total samples: {RV['Y'].shape[0]}")
    
    # Split people: 80% train, 10% test, 10% validation
    n_people_train = int(0.8 * n_people)
    n_people_test = int(0.1 * n_people)
    # Remaining goes to validation
    
    # Randomly shuffle person indices
    person_idxs = np.random.permutation(n_people)
    people_train = unique_people[person_idxs[:n_people_train]]
    people_test = unique_people[person_idxs[n_people_train:(n_people_train + n_people_test)]]
    people_val = unique_people[person_idxs[(n_people_train + n_people_test):]]
    
    print(f"\n📋 Split breakdown:")
    print(f"   Train: {len(people_train)} people × 9 views = {len(people_train) * 9} samples")
    print(f"   Test:  {len(people_test)} people × 9 views = {len(people_test) * 9} samples")
    print(f"   Val:   {len(people_val)} people × 9 views = {len(people_val) * 9} samples")
    
    # Create boolean masks based on person membership
    # Itrain: boolean array indicating which samples belong to training people
    Itrain = np.isin(RV["Did"], people_train)
    # Itest: boolean array indicating which samples belong to test people
    Itest = np.isin(RV["Did"], people_test)
    # Ival: boolean array indicating which samples belong to validation people
    Ival = np.isin(RV["Did"], people_val)
    
    # Verification: Each split should have all 9 views per person
    print(f"\n✅ Verification:")
    print(f"   Train samples: {np.sum(Itrain)} (expected: {len(people_train) * 9})")
    print(f"   Test samples:  {np.sum(Itest)} (expected: {len(people_test) * 9})")
    print(f"   Val samples:   {np.sum(Ival)} (expected: {len(people_val) * 9})")
    
    # Verify no overlap
    assert not np.any(Itrain & Itest), "Train and test overlap!"
    assert not np.any(Itrain & Ival), "Train and val overlap!"
    assert not np.any(Itest & Ival), "Test and val overlap!"
    print(f"   ✅ No person appears in multiple splits")
    
    # Check ethnicity distribution across splits
    # Extract ethnicity from Did (format: "ethnicity/PERSONID_SESSION")
    print(f"\n📊 Ethnicity distribution across splits:")
    
    def get_ethnicity_counts(did_array):
        """Extract ethnicity from person IDs and count them"""
        # Since Did format is "PERSONID_SESSION", we need to track which ethnicity folder they came from
        # This info isn't directly in Did, but we can infer from the file organization
        return len(np.unique(did_array))
    
    print(f"   Train unique people: {len(people_train)}")
    print(f"   Test unique people:  {len(people_test)}")
    print(f"   Val unique people:   {len(people_val)}")
    print(f"   Note: Ethnicity distribution depends on original folder organization")
    print(f"   With random seed(0), ethnicities should be mixed across splits")
    
    # Check ethnicity distribution in each split
    print(f"\n📊 ETHNICITY DISTRIBUTION BY SPLIT:")
    print(f"=" * 80)
    
    # Extract ethnicity from Did (format: "BF0601_1100" where BF = Black Female, etc.)
    # But actually, we need to track ethnicity from file paths
    # Since we don't have that info here, let's count unique people per split
    # and check view distribution instead
    
    # However, we can infer ethnicity from the person IDs if needed
    # For now, let's verify each person has all 9 views in their split
    
    print(f"\n🔍 Verifying person completeness in each split:")
    for split_name, mask in [("Train", Itrain), ("Test", Itest), ("Val", Ival)]:
        split_people = np.unique(RV["Did"][mask])
        split_rids = RV["Rid"][mask]
        
        # Check each person has all 9 views
        incomplete_people = []
        for person in split_people:
            person_mask = (RV["Did"] == person) & mask
            person_views = len(RV["Rid"][person_mask])
            if person_views != 9:
                incomplete_people.append((person.decode('utf-8'), person_views))
        
        if incomplete_people:
            print(f"   ❌ {split_name}: {len(incomplete_people)} people with incomplete views!")
            for pid, nviews in incomplete_people[:5]:  # Show first 5
                print(f"      {pid}: {nviews}/9 views")
        else:
            print(f"   ✅ {split_name}: All {len(split_people)} people have complete 9 views")
        
        # View distribution
        unique_views, view_counts = np.unique(split_rids, return_counts=True)
        view_dist = {v.decode('utf-8'): c for v, c in zip(unique_views, view_counts)}
        expected_per_view = len(split_people)
        
        all_equal = all(c == expected_per_view for c in view_counts)
        if all_equal:
            print(f"   ✅ {split_name}: All views have {expected_per_view} samples (balanced)")
        else:
            print(f"   ⚠️  {split_name}: Unbalanced view distribution:")
            for view, count in sorted(view_dist.items()):
                print(f"      {view}: {count} (expected: {expected_per_view})")

    out = {}
    # Split each dataset (Y, Did, Rid) into train/val/test
    for key in RV.keys():
        out["%s_train" % key] = RV[key][Itrain]
        out["%s_val" % key] = RV[key][Ival]
        out["%s_test" % key] = RV[key][Itest]

    return out


if __name__ == "__main__":

    main()
