import os
import json
import random

# Path configuration
dataset_root = "DATA/i-LIDS-VID/sequences"
cam1_dir = os.path.join(dataset_root, "cam1")
output_path = os.path.join(dataset_root, "splits.json")

def generate_splits():
    print(f"Checking directory: {cam1_dir}")
    
    if not os.path.exists(cam1_dir):
        print(f"CRITICAL ERROR: Could not find {cam1_dir}")
        return

    # 1. Get Person IDs
    pids = set()
    try:
        items = os.listdir(cam1_dir)
        for item in items:
            item_path = os.path.join(cam1_dir, item)
            # Check for folders like "person001"
            if os.path.isdir(item_path) and "person" in item:
                try:
                    pid_str = item.replace('person', '')
                    pid = int(pid_str)
                    pids.add(pid)
                except ValueError:
                    continue
    except Exception as e:
        print(f"Error reading directory: {e}")
        return

    pids = sorted(list(pids))
    print(f"Found {len(pids)} unique identities.")

    # 2. Generate 10 Random Splits
    num_splits = 10
    split_list = []
    
    # 50% for training, 50% for testing
    train_size = len(pids) // 2 
    
    print(f"Generating splits with 'trainval' key...")

    for i in range(num_splits):
        current_pids = list(pids)
        random.shuffle(current_pids)
        
        train_ids = current_pids[:train_size]
        test_ids = current_pids[train_size:]
        
        # KEY FIX HERE: The code expects 'trainval', 'query', and 'gallery'
        split_dict = {
            'trainval': train_ids,   # Was 'train', changed to 'trainval'
            'query': test_ids,       # The code likely needs these too
            'gallery': test_ids      # In split files, query/gallery usually share the same IDs
        }
        split_list.append(split_dict)

    # 3. Save to JSON
    with open(output_path, 'w') as f:
        json.dump(split_list, f, indent=4)
    
    print(f"Success! Saved fixed splits to: {output_path}")

if __name__ == "__main__":
    generate_splits()