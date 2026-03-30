import os
import os.path as osp
import re
from collections import defaultdict

import numpy as np

from utils.serialization import write_json, read_json


class AGReid(object):
    """
    AG-ReID: Aerial-Ground Person Re-Identification Dataset.

    Directory structure (flat, all images in one folder per split):
        DATA/AG-ReID/
            bounding_box_train/    P{pid}T{tid}A{ang}C{cam}F{frame}.jpg
            query_all_c0/          same filename format, camera 0 only
            query_all_c3/          same filename format, camera 3 only
            bounding_box_test_all_c0/  gallery, camera 0
            bounding_box_test_all_c3/  gallery, camera 3

    Camera mapping:  0 (ground) → 0,  3 (aerial) → 1
    Tracklet: all frames sharing the same (person_id, tracklet_id, camera_id)
    """

    FNAME_RE = re.compile(r'^P(\d+)T(\d+)A\d+C(\d+)F\d+\.jpg$')
    CAM_REMAP = {0: 0, 3: 1}

    def __init__(self, root='DATA/AG-ReID', *args, **kwargs):
        self._root = root

        self.split_train_json   = osp.join(root, 'split_train.json')
        self.split_query_json   = osp.join(root, 'split_query.json')
        self.split_gallery_json = osp.join(root, 'split_gallery.json')

        train_dirs   = [osp.join(root, 'bounding_box_train')]
        query_dirs   = [osp.join(root, 'query_all_c0'), osp.join(root, 'query_all_c3')]
        gallery_dirs = [osp.join(root, 'bounding_box_test_all_c0'),
                        osp.join(root, 'bounding_box_test_all_c3')]

        train, num_train_tracklets, num_train_pids, num_train_imgs, num_train_cams = \
            self._process_dirs(train_dirs, self.split_train_json, relabel=True)

        query, num_query_tracklets, num_query_pids, num_query_imgs, _ = \
            self._process_dirs(query_dirs, self.split_query_json, relabel=False)

        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs, _ = \
            self._process_dirs(gallery_dirs, self.split_gallery_json, relabel=False)

        all_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs
        print("=> AG-ReID loaded")
        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # tracklets | # imgs")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d}    | {:6d}".format(
            num_train_pids, num_train_tracklets, sum(num_train_imgs)))
        print("  query    | {:5d} | {:8d}    | {:6d}".format(
            num_query_pids, num_query_tracklets, sum(num_query_imgs)))
        print("  gallery  | {:5d} | {:8d}    | {:6d}".format(
            num_gallery_pids, num_gallery_tracklets, sum(num_gallery_imgs)))
        print("  ----------------------------------------")
        print("  imgs per tracklet: min={} max={} avg={:.1f}".format(
            min(all_imgs), max(all_imgs), np.mean(all_imgs)))
        print("  num cameras (train): {}".format(num_train_cams))
        print("  ----------------------------------------")

        self.train   = train
        self.query   = query
        self.gallery = gallery

        self.num_train_pids    = num_train_pids
        self.num_query_pids    = num_query_pids
        self.num_gallery_pids  = num_gallery_pids
        self.num_train_cams    = num_train_cams
        self.num_train_vids    = 1

    def _parse_fname(self, fname):
        """Return (pid_int, tid_str, cam_remapped) or None if not a valid image."""
        m = self.FNAME_RE.match(fname)
        if m is None:
            return None
        pid  = int(m.group(1))
        tid  = m.group(2)          # keep as string for grouping key
        cam  = int(m.group(3))
        cam_r = self.CAM_REMAP.get(cam, cam)
        return pid, tid, cam_r

    def _process_dirs(self, dirs, json_path, relabel):
        if osp.exists(json_path):
            split = read_json(json_path)
            return (split['tracklets'], split['num_tracklets'], split['num_pids'],
                    split['num_imgs_per_tracklet'], split['num_cams'])

        # group images by (pid, tid, cam)
        groups = defaultdict(list)
        for d in dirs:
            for fname in sorted(os.listdir(d)):
                parsed = self._parse_fname(fname)
                if parsed is None:
                    continue          # skip Zone.Identifier and other junk
                pid, tid, cam_r = parsed
                groups[(pid, tid, cam_r)].append(osp.join(d, fname))

        pid_set = sorted({k[0] for k in groups})
        if relabel:
            pid2label = {pid: label for label, pid in enumerate(pid_set)}

        cams_seen = set()
        tracklets = []
        num_imgs_per_tracklet = []

        for (pid, tid, cam_r), img_paths in sorted(groups.items()):
            img_paths.sort()
            cams_seen.add(cam_r)
            label = pid2label[pid] if relabel else pid
            tracklets.append((img_paths, label, cam_r, 1))
            num_imgs_per_tracklet.append(len(img_paths))

        split = {
            'tracklets':              tracklets,
            'num_tracklets':          len(tracklets),
            'num_pids':               len(pid_set),
            'num_imgs_per_tracklet':  num_imgs_per_tracklet,
            'num_cams':               len(cams_seen),
            'num_tracks':             1,
        }
        print("Saving split to {}".format(json_path))
        write_json(split, json_path)

        return tracklets, len(tracklets), len(pid_set), num_imgs_per_tracklet, len(cams_seen)
