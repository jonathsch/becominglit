import json
from collections import OrderedDict
from pathlib import Path
from typing import List

import numpy as np
import torch
from torchvision.utils import draw_keypoints

from becominglit.util.env import BECOMINGLIT_DATASET_PATH

CAM_SERIALS = [
    "222200042",
    "222200044",
    "222200046",
    "222200040",
    "222200036",
    "222200048",
    "220700191",
    "222200041",
    "222200037",
    "222200038",
    "222200047",
    "222200043",
    "222200049",
    "222200039",
    "222200045",
    "221501007",
]

DLIB_68_TO_WFLW_98_IDX_MAPPING = OrderedDict()
DLIB_68_TO_WFLW_98_IDX_MAPPING.update(dict(zip(range(0, 17), range(0, 34, 2))))  # jaw | 17 pts
DLIB_68_TO_WFLW_98_IDX_MAPPING.update(dict(zip(range(17, 22), range(33, 38))))  # left upper eyebrow points | 5 pts
DLIB_68_TO_WFLW_98_IDX_MAPPING.update(dict(zip(range(22, 27), range(42, 47))))  # right upper eyebrow points | 5 pts
DLIB_68_TO_WFLW_98_IDX_MAPPING.update(dict(zip(range(27, 36), range(51, 60))))  # nose points | 9 pts
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({36: 60})  # left eye points | 6 pts
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({37: 61})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({38: 63})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({39: 64})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({40: 65})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({41: 67})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({42: 68})  # right eye | 6 pts
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({43: 69})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({44: 71})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({45: 72})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({46: 73})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({47: 75})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update(dict(zip(range(48, 68), range(76, 96))))  # mouth points | 20 pts

ibu68_index_into_wflw = []
for i in range(68):
    ibu68_index_into_wflw.append(DLIB_68_TO_WFLW_98_IDX_MAPPING[i])
ibu68_index_into_wflw = np.array(ibu68_index_into_wflw)


def get_fully_lit_frames(sid: str, sequence: str) -> List[int]:
    """Get list of fully lit frames for a given subject and sequence.

    Args:
        sid (str): Subject ID.
        seq (str): Sequence name.

    Returns:
        List[int]: List of fully lit frame indices.
    """
    with open(
        Path(BECOMINGLIT_DATASET_PATH).joinpath(sid, "sequences", sequence, "light_pattern_per_frame.json"), "r"
    ) as f:
        light_patterns = json.load(f)

    frame_list = [frame_id for (frame_id, pattern_idx) in light_patterns if pattern_idx == 0]
    return frame_list


def list_sequences(sid: str) -> List[str]:
    """List all sequences for a given subject.

    Args:
        sid (str): Subject ID.

    Returns:
        List[str]: List of sequence names.
    """
    dataset_path = Path(BECOMINGLIT_DATASET_PATH)
    subject_path = dataset_path.joinpath(sid, "sequences")
    sequences = [
        seq_dir.name
        for seq_dir in subject_path.iterdir()
        if seq_dir.is_dir() and not seq_dir.name == "BACKGROUND" and "CALIB" not in seq_dir.name
    ]
    return sequences


def normalize_image_points(u, v, resolution):
    """
    normalizes u, v coordinates from [0 ,image_size] to [-1, 1]
    :param u:
    :param v:
    :param resolution:
    :return:
    """
    u = 2 * (u - resolution[1] / 2.0) / resolution[1]
    v = 2 * (v - resolution[0] / 2.0) / resolution[0]
    return u, v


def plot_landmarks_2d(
    img: torch.Tensor,
    lmks: torch.Tensor,
    connectivity=None,
    colors="white",
    unit=1,
    input_float=False,
):
    if input_float:
        img = (img * 255).byte()

    try:
        img = draw_keypoints(
            img,
            lmks,
            connectivity=connectivity,
            colors=colors,
            radius=2 * unit,
            width=2 * unit,
        )
    except Exception:
        pass

    if input_float:
        img = img.float() / 255
    return img
