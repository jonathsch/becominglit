from argparse import ArgumentParser
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
from transformers import AutoModelForImageSegmentation

from becominglit.data.preprocess.utils import CAM_SERIALS, get_fully_lit_frames, list_sequences
from becominglit.util.env import BECOMINGLIT_DATASET_PATH, IMAGE_FILE_FORMAT
from becominglit.util.image import scale_img_nchw


@torch.inference_mode()
def main(subject: str, sequence: str):
    device = torch.device("cuda:0")
    fully_lit_frames = get_fully_lit_frames(subject, sequence)

    birefnet = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet_HR-matting", trust_remote_code=True)
    birefnet = birefnet.to(device)
    birefnet.eval()

    sequence_path = Path(BECOMINGLIT_DATASET_PATH).joinpath(subject, "sequences", sequence)

    for cid in sorted(CAM_SERIALS):
        # Make output folder
        alpha_cam_folder = sequence_path.joinpath("alpha_birefnet", f"cam_{cid}")
        alpha_cam_folder.mkdir(parents=True, exist_ok=True)

        for timestep in tqdm(fully_lit_frames, desc=f"Processing cam {cid}"):
            # Load image
            img_path = sequence_path.joinpath("images", f"cam_{cid}", f"frame_{timestep:06d}.{IMAGE_FILE_FORMAT}")
            img = to_tensor(Image.open(img_path)).to(device)[None]  # [1, 3, H, W]
            img_scaled = scale_img_nchw(img, (2048, 2048))

            # Segment background
            preds = birefnet(img_scaled)[-1].sigmoid()
            preds = scale_img_nchw(preds, size=img.shape[-2:])

            # Save alpha mask
            pred_np = preds[0].mul(255).byte().permute(1, 2, 0).cpu().numpy().squeeze(-1)  # [H, W]
            alpha_path = alpha_cam_folder.joinpath(f"frame_{timestep:06d}.jpg")
            Image.fromarray(pred_np).save(alpha_path)


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate alpha masks using BiRefNet")
    parser.add_argument("sid", type=str, help="Subject ID")
    args = parser.parse_args()

    sid = args.sid
    sequences = sorted(list_sequences(sid))

    for seq in sequences:
        print(f"Processing subject {sid}, sequence {seq}...")
        main(sid, seq)
