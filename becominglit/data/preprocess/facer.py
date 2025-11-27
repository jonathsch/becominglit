from argparse import ArgumentParser
from pathlib import Path

import facer
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from becominglit.data.preprocess.utils import CAM_SERIALS, get_fully_lit_frames, list_sequences
from becominglit.util.env import BECOMINGLIT_DATASET_PATH, IMAGE_FILE_FORMAT
from becominglit.util.image import scale_img_hwc


@torch.inference_mode()
def main(subject: str, sequence: str):
    device = torch.device("cuda:0")

    img_width = 2200
    img_height = 3208

    fully_lit_frames = get_fully_lit_frames(subject, sequence)

    face_parser = facer.face_parser("farl/celebm/448", device=device)

    sequence_path = Path(BECOMINGLIT_DATASET_PATH).joinpath(subject, "sequences", sequence)

    for cid in CAM_SERIALS:
        # Make output folder
        facer_cam_folder = sequence_path.joinpath("facer_segmentation", f"cam_{cid}")
        facer_cam_folder.mkdir(parents=True, exist_ok=True)

        for idx_frame, frame_id in enumerate(tqdm(fully_lit_frames, desc=f"Processing cam {cid}")):
            img_path = sequence_path.joinpath("images", f"cam_{cid}", f"frame_{frame_id:06d}.{IMAGE_FILE_FORMAT}")
            img = facer.read_hwc(img_path).to(device=device, dtype=torch.float32)  # [H, W, 3]
            img = scale_img_hwc(img, size=(img_height // 2, img_width // 2))
            img = facer.hwc2bchw(facer.read_hwc(img_path)).to(device=device)  # image: 1 x 3 x h x w

            seg_logits, seg_preds, label_names = face_parser.forward_warped(img)

            mask = seg_preds[0].cpu().numpy().astype(np.uint8)  # [H, W]
            mask = Image.fromarray(mask).resize((img_width, img_height), resample=Image.Resampling.NEAREST)
            mask.save(facer_cam_folder.joinpath(f"frame_{frame_id:06d}.png"))


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate segmentation masks using Facer")
    parser.add_argument("sid", type=str, help="Subject ID")
    args = parser.parse_args()

    sid = args.sid
    sequences = sorted(list_sequences(sid))

    for seq in sequences:
        print(f"Processing subject {sid}, sequence {seq}...")
        main(sid, seq)
