import os
import shutil
import random

def copy_subfolders(root_folder, target_a, target_b, percentage = 80):
    # Check if targets exist, create them if not
    for target in [target_a, target_b]:
        if not os.path.exists(target):
            os.makedirs(target)

    # List all subfolders
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]

    # Shuffle the list to randomize selection
    random.shuffle(subfolders)

    # Calculate the number of subfolders to copy to target_a
    num_a = int(len(subfolders) * (percentage / 100))
    num_b = len(subfolders) - num_a

    # Copy subfolders to target_a and target_b
    for folder in subfolders[:num_a]:
        shutil.copytree(folder, os.path.join(target_a, os.path.basename(folder)))
    for folder in subfolders[num_a:]:
        shutil.copytree(folder, os.path.join(target_b, os.path.basename(folder)))
