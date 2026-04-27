import os
import re
import json

AERIAL_CAMS = {4, 5}   # C4, C5 = drone cameras
GROUND_CAMS = {0, 1, 2, 3}  # C0-C3 = CCTV / wearable

FNAME_RE = re.compile(r'C(\d+)')


def _cam_from_dir(img_dir):
    """Return camera int from first jpg in directory, or None."""
    for f in os.listdir(img_dir):
        if f.endswith('.jpg'):
            m = FNAME_RE.search(f)
            if m:
                return int(m.group(1))
    return None


def _load_split(split_dir, cache_path=None):
    """Load all tracklets from split_dir/{pid}/tracklet_*/ structure.
    Caches result to cache_path JSON to avoid rescanning on subsequent runs."""
    if cache_path and os.path.exists(cache_path):
        print(f"=> Loading scan cache: {cache_path}")
        with open(cache_path) as f:
            raw = json.load(f)
        return [(entry[0], entry[1], entry[2], entry[3]) for entry in raw]

    print(f"=> Scanning {split_dir} ...")
    tracklets = []
    for pid_str in sorted(os.listdir(split_dir), key=lambda x: int(x) if x.isdigit() else x):
        pid_dir = os.path.join(split_dir, pid_str)
        if not os.path.isdir(pid_dir):
            continue
        pid_int = int(pid_str)
        for t_name in sorted(os.listdir(pid_dir)):
            t_dir = os.path.join(pid_dir, t_name)
            if not os.path.isdir(t_dir):
                continue
            imgs = sorted([os.path.join(t_dir, f)
                           for f in os.listdir(t_dir)
                           if f.endswith('.jpg')])
            if not imgs:
                continue
            cam = _cam_from_dir(t_dir)
            if cam is None:
                continue
            tracklets.append((imgs, pid_int, cam, 1))

    if cache_path:
        with open(cache_path, 'w') as f:
            json.dump(tracklets, f)
        print(f"=> Scan cache saved: {cache_path}")

    return tracklets


class AGVPReID(object):
    """
    AG-VPReID: Aerial-Ground Video Person Re-Identification (full dataset).

    Directory layout:
        DATA/AG-VPReID/
            train/{pid}/tracklet_*/
            case1_aerial_to_ground/
                query/{pid}/tracklet_*/    (aerial)
                gallery/{pid}/tracklet_*/  (ground)
            case2_ground_to_aerial/
                query/{pid}/tracklet_*/    (ground)
                gallery/{pid}/tracklet_*/  (aerial)

    self.train   — all training tracklets (relabelled 0..n_train_pids-1)
    self.query   — case1 query (aerial)
    self.gallery — case1 gallery (ground)
    self.case2_query   — case2 query (ground)
    self.case2_gallery — case2 gallery (aerial)

    Each tracklet entry: (img_paths_list, pid_label, cam_id, 1)
    """

    def __init__(self, root='DATA/AG-VPReID', max_pids=0, *args, **kwargs):
        train_dir = os.path.join(root, 'train')
        case1_dir = os.path.join(root, 'case1_aerial_to_ground')
        case2_dir = os.path.join(root, 'case2_ground_to_aerial')

        # --- training set (relabelled) ---
        raw_train = _load_split(train_dir,
                                cache_path=os.path.join(root, 'scan_cache_train.json'))
        all_pids = sorted({t[1] for t in raw_train})
        n_total_train_pids = len(all_pids)
        if max_pids and max_pids < n_total_train_pids:
            print(f"=> Subsetting to {max_pids}/{n_total_train_pids} train IDs (DATASETS.MAX_PIDS)")
            all_pids = all_pids[:max_pids]
        pid2label = {p: i for i, p in enumerate(all_pids)}
        train_tracklets = [(imgs, pid2label[pid], cam, seq)
                           for imgs, pid, cam, seq in raw_train
                           if pid in pid2label]

        # --- eval splits (original pid ints as labels) ---
        case1_query   = _load_split(os.path.join(case1_dir, 'query'),
                                    cache_path=os.path.join(root, 'scan_cache_c1q.json'))
        case1_gallery = _load_split(os.path.join(case1_dir, 'gallery'),
                                    cache_path=os.path.join(root, 'scan_cache_c1g.json'))

        # --- proportional eval subset ---
        if max_pids and max_pids < n_total_train_pids:
            all_c1_pids = sorted({t[1] for t in case1_query})
            n_eval = round(max_pids * len(all_c1_pids) / n_total_train_pids)
            keep_eval = set(all_c1_pids[:n_eval])
            case1_query   = [t for t in case1_query   if t[1] in keep_eval]
            case1_gallery = [t for t in case1_gallery if t[1] in keep_eval]
            print(f"=> Subsetting eval to {n_eval}/{len(all_c1_pids)} case1 IDs")

        case2_missing = not os.path.isdir(case2_dir)
        if case2_missing:
            print(f"WARNING: case2 directory not found ({case2_dir}). "
                  "case2_query and case2_gallery will be empty. "
                  "Eval with --case 2 or --case 0 (both) will be skipped/invalid.")
            case2_query, case2_gallery = [], []
        else:
            case2_query   = _load_split(os.path.join(case2_dir, 'query'),
                                        cache_path=os.path.join(root, 'scan_cache_c2q.json'))
            case2_gallery = _load_split(os.path.join(case2_dir, 'gallery'),
                                        cache_path=os.path.join(root, 'scan_cache_c2g.json'))

        n_train_cams = len({t[2] for t in train_tracklets})

        print("=> AG-VPReID loaded")
        print("  train  : {:4d} ids | {:5d} tracklets | {:d} cameras".format(
            len(pid2label), len(train_tracklets), n_train_cams))
        print("  case1 query  : {:4d} ids | {:5d} tracklets (aerial)".format(
            len({t[1] for t in case1_query}), len(case1_query)))
        print("  case1 gallery: {:4d} ids | {:5d} tracklets (ground)".format(
            len({t[1] for t in case1_gallery}), len(case1_gallery)))
        if case2_missing:
            print("  case2        : NOT AVAILABLE (directory missing)")
        else:
            print("  case2 query  : {:4d} ids | {:5d} tracklets (ground)".format(
                len({t[1] for t in case2_query}), len(case2_query)))
            print("  case2 gallery: {:4d} ids | {:5d} tracklets (aerial)".format(
                len({t[1] for t in case2_gallery}), len(case2_gallery)))

        self.train = train_tracklets
        self.query = case1_query
        self.gallery = case1_gallery
        self.case2_query = case2_query
        self.case2_gallery = case2_gallery

        self.num_train_pids = len(pid2label)
        self.num_query_pids = len({t[1] for t in case1_query})
        self.num_gallery_pids = len({t[1] for t in case1_gallery})
        self.num_train_cams = n_train_cams
        self.num_train_vids = 1
