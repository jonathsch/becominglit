import json
import logging
import shutil
from multiprocessing import Pool
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import tyro

from becominglit.util.env import BECOMINGLIT_DATASET_PATH

IMG_WIDTH = 2200
IMG_HEIGHT = 3208
ALPHA_FOLDER_NAME = "alpha_birefnet"
SEGMENTATION_FOLDER_NAME = "facer_segmentation"
NUM_PROCESSES = 8

logger = logging.Logger(__name__, level=logging.INFO)


class MaskInterpolator:
    def __init__(
        self,
        subject: str,
        sequence: str,
        /,
    ):
        self.data_root = Path(BECOMINGLIT_DATASET_PATH)
        self.subject = subject
        self.sequence = sequence

        self.subject_folder = self.data_root / subject
        self.seq_folder = self.data_root / subject / "sequences" / sequence

        self.alpha_folder = self.seq_folder / ALPHA_FOLDER_NAME
        self.segmentation_folder = self.seq_folder / SEGMENTATION_FOLDER_NAME

        # Get non-tracking frames
        with open(self.seq_folder / "light_pattern_per_frame.json", "r") as f:
            light_pattern = json.load(f)
        self.tracking_frames = [lp[0] for lp in light_pattern if lp[1] == 0]
        self.frame_list = [lp[0] for lp in light_pattern if lp[1] != 0]

        self.timestep_ids = sorted(self.frame_list)

        logger.info(
            f"Found {len(self.frame_list + self.tracking_frames)} frames. {len(self.tracking_frames)} tracking frames, {len(self.frame_list)} OLAT frames"
        )

        # Get camera ids
        with open(self.subject_folder / "calibration" / "camera_calibration.json") as f:
            cam_calib = json.load(f)
        self.cam_ids = list(cam_calib["world_to_cam"].keys())

    @property
    def num_frames(self):
        return len(self.frame_list)

    def get_nearest_tracking_frames(self, tid):
        if tid < self.tracking_frames[0]:
            return self.tracking_frames[0], self.tracking_frames[0]
        elif tid > self.tracking_frames[-1]:
            return self.tracking_frames[-1], self.tracking_frames[-1]
        elif tid in self.tracking_frames:
            return tid, tid
        elif tid - 1 in self.tracking_frames:
            return tid - 1, tid + 2
        elif tid - 2 in self.tracking_frames:
            return tid - 2, tid + 1
        else:
            raise ValueError(f"Invalid timestep {tid}")

    def get_nearest_tracking_frame(self, tid):
        if tid < self.tracking_frames[0]:
            return self.tracking_frames[0]
        elif tid > self.tracking_frames[-1]:
            return self.tracking_frames[-1]
        elif tid in self.tracking_frames:
            return tid
        elif tid - 1 in self.tracking_frames:
            return tid - 1
        elif tid + 1 in self.tracking_frames:
            return tid + 1
        else:
            raise ValueError(f"Invalid timestep {tid}")

    def export_alpha_mask(self, timestep_id):
        for cid in self.cam_ids:
            # Skip if file exists
            save_folder = self.alpha_folder / f"cam_{cid}"

            # Load alpha mask from bg matting
            track_frame_before, track_frame_after = self.get_nearest_tracking_frames(timestep_id)

            # Make union of tracking frame alpha masks
            alpha_bg_track_before_path = self.alpha_folder / f"cam_{cid}" / f"frame_{track_frame_before:06d}.jpg"
            alpha_bg_track_after_path = self.alpha_folder / f"cam_{cid}" / f"frame_{track_frame_after:06d}.jpg"

            alpha_bg_track_before = iio.imread(alpha_bg_track_before_path)
            alpha_bg_track_after = iio.imread(alpha_bg_track_after_path)
            alpha_bg_track_union = np.maximum(alpha_bg_track_before, alpha_bg_track_after)

            iio.imwrite(
                save_folder / f"frame_{timestep_id:06d}.jpg",
                alpha_bg_track_union,
                quality=95,
            )

    def export_seg_mask(self, timestep_id):
        for cid in self.cam_ids:
            save_folder = self.segmentation_folder / f"cam_{cid}"

            # Load closest segmentation mask from bg matting
            seg_frame_id = self.get_nearest_tracking_frame(timestep_id)
            seg_frame_path = self.seq_folder / SEGMENTATION_FOLDER_NAME / f"cam_{cid}" / f"frame_{seg_frame_id:06d}.png"
            seq_frame_tgt_path = save_folder / f"frame_{timestep_id:06d}.png"

            shutil.copy2(seg_frame_path, seq_frame_tgt_path)

    def export(self):
        with Pool(processes=NUM_PROCESSES) as pool:
            pool.map(self.export_alpha_mask, self.timestep_ids)
            pool.map(self.export_seg_mask, self.timestep_ids)


if __name__ == "__main__":
    exporter = tyro.cli(MaskInterpolator)
    exporter.export()
