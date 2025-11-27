from argparse import ArgumentParser
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torchlm
from torchlm.models import pipnet
from torchlm.tools import faceboxesv2
from tqdm import tqdm

from becominglit.data.preprocess.utils import CAM_SERIALS, get_fully_lit_frames, list_sequences
from becominglit.util.env import BECOMINGLIT_DATASET_PATH, IMAGE_FILE_FORMAT


@torch.inference_mode()
def main(subject: str, sequence: str, normalize: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup model
    torchlm.runtime.bind(faceboxesv2(device=device))  # set device="cuda" if you want to run with CUDA
    torchlm.runtime.bind(
        pipnet(
            backbone="resnet18",
            pretrained=True,
            num_nb=10,
            num_lms=98,
            net_stride=32,
            input_size=256,
            meanface_type="wflw",
            map_location=device,
            checkpoint=None,
        )
    )  # will auto download pretrained weights from latest release if pretrained=True

    fully_lit_frames = get_fully_lit_frames(subject, sequence)

    sequence_path = Path(BECOMINGLIT_DATASET_PATH).joinpath(subject, "sequences", sequence)

    for cid in CAM_SERIALS:
        landmark_dict = {}
        bbox_dict = {}

        for tid in tqdm(fully_lit_frames, desc=f"Processing cam {cid}"):
            img_path = sequence_path.joinpath("images", f"cam_{cid}", f"frame_{tid:06d}.{IMAGE_FILE_FORMAT}")
            image = iio.imread(img_path)

            landmarks, bboxes = torchlm.runtime.forward(image)

            if landmarks.shape != (1, 98, 2):
                print(f"Landmarks shape mismatch for {tid} in cam {cid}: {landmarks.shape}")
                continue

            if bboxes.shape != (1, 5):
                print(f"BBoxes shape mismatch for {tid} in cam {cid}: {bboxes.shape}")
                continue

            landmarks = landmarks[0]
            bboxes = bboxes[0]

            # Normalize landmarks to [0, 1]
            if normalize:
                landmarks[:, 0] /= image.shape[1]
                landmarks[:, 1] /= image.shape[0]

                bboxes[0] /= image.shape[1]
                bboxes[1] /= image.shape[0]
                bboxes[2] /= image.shape[1]
                bboxes[3] /= image.shape[0]

            landmark_dict[str(tid)] = landmarks
            bbox_dict[str(tid)] = bboxes

        # Save the landmarks and bboxes
        lmk_save_path = sequence_path.joinpath("landmark2d", "PIPnet", f"cam_{cid}.npz")
        lmk_save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            lmk_save_path,
            **landmark_dict,
        )

        bbox_save_path = sequence_path.joinpath("bbox", "PIPnet", f"cam_{cid}.npz")
        bbox_save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            bbox_save_path,
            **bbox_dict,
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate 2D landmarks using PIPNet")
    parser.add_argument("sid", type=str, help="Subject ID")
    args = parser.parse_args()

    sid = args.sid
    sequences = sorted(list_sequences(sid))

    for seq in sequences:
        print(f"Processing subject {sid}, sequence {seq}...")
        main(sid, seq)
