import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, List, Literal, Optional

import numpy as np
import pillow_avif  # noqa: F401
import plyfile
import torch
import torch.nn.functional as thf
from PIL import Image
from torch.utils.data import Dataset, default_collate
from torchvision.transforms.functional import pil_to_tensor

from becominglit.util.env import BECOMINGLIT_DATASET_PATH, IMAGE_FILE_FORMAT
from becominglit.util.image import color_correct_srgb, scale_img_nchw, srgb2linear, srgb2linear_cc

logger = logging.getLogger(__name__)

NUM_CAMS = 16
MAX_NUM_SEQUENCES = 15


class BecomingLitDataset(Dataset):
    def __init__(
        self,
        subject: str,
        downscale_factor: int = 2,
        fully_lit_only: bool = False,
        partially_lit_only: bool = False,
        apply_color_correction: bool = True,
        apply_alpha: bool = True,
        black_level_subtraction: float = 0.0,
        color_space: Literal["linear", "sRGB"] = "linear",
        sequences_subset: Optional[Iterable[str]] = None,
        cameras_subset: Optional[Iterable[str]] = None,
        light_pattern_subset: Optional[Iterable[int]] = None,
        length_unit: Literal["m", "mm"] = "m",
    ):
        super().__init__()
        self.base_path = Path(BECOMINGLIT_DATASET_PATH) / str(subject)
        self.subject = str(subject)
        self.downscale_factor = downscale_factor
        self.fully_lit_only = fully_lit_only
        self.partially_lit_only = partially_lit_only
        self.apply_color_correction = apply_color_correction
        self.apply_alpha = apply_alpha
        self.black_level_subtraction = black_level_subtraction
        self.color_space = color_space
        self.length_unit = length_unit

        self.sequences_subset = set(sequences_subset or {})
        self.sequences = list(self.get_sequences())

        self.cameras_subset = set(cameras_subset or {})
        self.cameras = list(self.get_camera_calibration().keys())

        self.light_pattern_subset = set(light_pattern_subset or {})
        self.light_pattern_subset = set(map(int, self.light_pattern_subset))

        # create linear index (seq, frame, cam_id) for all items
        self.items = []
        for seq in self.sequences:
            for frame in self.get_frame_list(seq, fully_lit_only, partially_lit_only):
                for cam_id in self.cameras:
                    self.items.append((seq, frame, cam_id))

    @lru_cache(maxsize=1)
    def get_sequences(self) -> List[str]:
        sequences = [
            p.name
            for p in self.base_path.joinpath("sequences").iterdir()
            if p.is_dir() and not p.name.startswith("CALIB") and not p.name.startswith("BACK")
        ]
        logger.info(f"Found {len(sequences)} available sequences.")

        if self.sequences_subset:
            sequences = list(filter(lambda s: s in self.sequences_subset, sequences))
            logger.info(f"Left with {len(sequences)} sequences after filtering for passed sequence subset")

        return sequences

    @lru_cache(maxsize=1)
    def get_camera_calibration(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        with open(self.base_path.joinpath("calibration", "camera_calibration.json"), "r") as f:
            cam_calib = json.load(f)

        # intrinsics
        K = torch.tensor(
            [
                [cam_calib["cam_data"]["fx"], 0, cam_calib["cam_data"]["cx"]],
                [0, cam_calib["cam_data"]["fy"], cam_calib["cam_data"]["cy"]],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )
        K[:2, :] /= self.downscale_factor  # Downscale intrinsics (pixel units)

        # World to camera matrices
        w2c = {cid: torch.tensor(Rt) for (cid, Rt) in cam_calib["world_to_cam"].items()}

        # Convert to millimeters if necessary
        if self.length_unit == "mm":
            for _, Rt in w2c.items():
                Rt[:3, 3] *= 1000.0

        if self.cameras_subset:
            w2c = {cid: w2c for cid, w2c in w2c.items() if cid in self.cameras_subset}
            logger.info(f"Left with {len(w2c)} cameras after filtering for passed camera subset")

        return {cid: (K, w2c) for cid, w2c in w2c.items()}

    @lru_cache(maxsize=NUM_CAMS)
    def get_camera_parameters(self, cam_id: str) -> dict[str, Any]:
        K, w2c = self.get_camera_calibration()[cam_id]
        R, t = w2c[:3, :3], w2c[:3, 3]
        focal = K[:2, :2]
        princpt = K[:2, 2]

        return {
            "Rt": w2c,
            "K": K,
            "campos": R.T @ -t,
            "camrot": R,
            "focal": focal,
            "princpt": princpt,
            "camera_id": cam_id,
        }

    @lru_cache(maxsize=1)
    def get_color_correction(self):
        with open(self.base_path / "calibration" / "color_calibration.json", "r") as f:
            color_correction = json.load(f)

        ccm_dict = {cid: torch.tensor(ccm) for cid, ccm in color_correction.items()}
        return ccm_dict

    def construct_batched_ccm(self, cam_ids: list[str]):
        ccm_dict = self.get_color_correction()
        batched_ccm = torch.stack([ccm_dict[cid] for cid in cam_ids], dim=0)
        return batched_ccm

    @lru_cache(maxsize=2 * MAX_NUM_SEQUENCES)
    def get_frame_list(self, sequence, fully_lit_only: bool = False, partially_lit_only: bool = False) -> list[int]:
        assert not (fully_lit_only and partially_lit_only), (
            "Cannot filter for fully lit and partially lit frames at the same time."
        )

        with open(self.base_path / "sequences" / sequence / "light_pattern_per_frame.json", "r") as f:
            light_patterns = json.load(f)

        if fully_lit_only:
            frame_list = [frame_id for (frame_id, pattern_idx) in light_patterns if pattern_idx == 0]
        elif partially_lit_only:
            frame_list = [frame_id for (frame_id, pattern_idx) in light_patterns if pattern_idx != 0]
        else:
            frame_list = [frame_id for (frame_id, pattern_idx) in light_patterns]

        return frame_list

    def load_assets(self):
        mean_texture = Image.open(self.base_path.joinpath("assets", "color_mean.png"))
        mean_texture = torch.as_tensor(np.array(mean_texture), dtype=torch.float32) / 255.0

        # light_positions
        light_meta_path = self.base_path / "calibration" / "light_pattern_metadata.json"
        with open(light_meta_path, "r") as f:
            light_meta_path = json.load(f)
        light_positions = torch.as_tensor(light_meta_path["light_positions"], dtype=torch.float32)  # [N, 3]

        # irradiance map
        fg_lut = torch.as_tensor(
            np.fromfile("assets/irrmaps/bsdf_256_256.bin", dtype=np.float32).reshape(1, 256, 256, 2),
            dtype=torch.float32,
        )

        return {
            # "color_mean": torch.as_tensor(np.array(mean_texture), dtype=torch.float32),
            "light_positions": light_positions,
            "fg_lut": fg_lut,
            "mean_texture": mean_texture,
        }

    def load_image(self, sequence: str, timestep: int, cam_id: str) -> torch.Tensor:
        img_path = (
            self.base_path
            / "sequences"
            / sequence
            / "images"
            / f"cam_{cam_id}"
            / f"frame_{timestep:06d}.{IMAGE_FILE_FORMAT}"
        )
        return pil_to_tensor(Image.open(img_path))

    def load_alpha(self, sequence: str, timestep: int, cam_id: str) -> torch.Tensor:
        alpha_path = (
            self.base_path / "sequences" / sequence / "alpha_birefnet" / f"cam_{cam_id}" / f"frame_{timestep:06d}.jpg"
        )
        return pil_to_tensor(Image.open(alpha_path))

    def load_segmentation(self, sequence: str, timestep: int, cam_id: str) -> torch.Tensor:
        seg_path = (
            self.base_path
            / "sequences"
            / sequence
            / "facer_segmentation"
            / f"cam_{cam_id}"
            / f"frame_{timestep:06d}.png"
        )
        return pil_to_tensor(Image.open(seg_path))

    def load_flame_vertices(self, sequence: str, timestep: int) -> torch.Tensor:
        flame_ply_path = (
            self.base_path / "sequences" / sequence / "flame_tracking/meshes_cano" / f"frame_{timestep:06d}.ply"
        )
        with open(flame_ply_path, "rb") as f:
            ply_data = plyfile.PlyData.read(f)
        vertices = np.vstack([ply_data["vertex"]["x"], ply_data["vertex"]["y"], ply_data["vertex"]["z"]]).T
        vertices = torch.as_tensor(vertices, dtype=torch.float32)
        if self.length_unit == "mm":
            vertices *= 1000.0
        return vertices

    def load_flame_params(self, sequence: str, timestep: int) -> dict[str, torch.Tensor]:
        flame_param_path = (
            self.base_path / "sequences" / sequence / "flame_tracking/flame_params" / f"frame_{timestep:06d}.npz"
        )
        flame_params = np.load(flame_param_path)
        flame_dict = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in flame_params.items()}
        flame_dict["shape"] = flame_dict["shape"].squeeze(0)
        flame_dict["static_offset"] = flame_dict["static_offset"].squeeze(0)
        if self.length_unit == "mm":
            flame_dict["translation"] = flame_dict["translation"].squeeze(0) * 1000.0
        return flame_dict

    @lru_cache(maxsize=MAX_NUM_SEQUENCES)
    def load_light_pattern(self, sequence: str) -> list[tuple[int]]:
        light_pattern_path = self.base_path / "sequences" / sequence / "light_pattern_per_frame.json"
        with open(light_pattern_path, "r") as f:
            return json.load(f)

    @lru_cache(maxsize=MAX_NUM_SEQUENCES)
    def load_light_pattern_meta(self, sequence: str) -> dict[str, Any]:
        light_pattern_path = self.base_path / "calibration" / "light_pattern_metadata.json"
        with open(light_pattern_path, "r") as f:
            return json.load(f)

    def load_light_info(self, sequence: str, timestep: int):
        is_fully_lit_frame: bool = timestep in self.get_frame_list(sequence, fully_lit_only=True)

        light_pattern = self.load_light_pattern(sequence)
        light_pattern = {f[0]: f[1] for f in light_pattern}
        light_pattern_meta = self.load_light_pattern_meta(sequence)
        light_pos_all = torch.FloatTensor(light_pattern_meta["light_positions"])
        n_lights_all = light_pos_all.shape[0]

        lightinfo = torch.IntTensor(
            light_pattern_meta["light_pattern"][light_pattern[timestep]]["light_index_durations"]
        )

        n_lights = lightinfo.shape[0]
        light_pos = light_pos_all[lightinfo[:, 0]]
        light_intensity = lightinfo[:, 1:].float() / 3000.0
        light_intensity = light_intensity * (1.0 / 0.66) if not is_fully_lit_frame else light_intensity
        light_pos = thf.pad(light_pos, (0, 0, 0, n_lights_all - n_lights), "constant", 0)
        light_intensity = thf.pad(light_intensity, (0, 0, 0, n_lights_all - n_lights), "constant", 0).repeat(1, 3)

        if self.length_unit == "mm":
            light_pos *= 1000.0

        return {
            "is_fullylit_frame": is_fully_lit_frame,
            "light_pos": light_pos,
            "light_intensity": light_intensity,
            "n_lights": n_lights,
        }

    @torch.no_grad()
    def batch_transform_fn(self, batch):
        # NOTE: We defer all compute-heavy data operations like downsampling, color correction, etc. to when the data is on the GPU.
        # I.e., this function is supposed to be called on the object returned by the dataloader."""
        device = batch["image"].device
        h, w = batch["image"].shape[-2:]
        h = int(h // self.downscale_factor)
        w = int(w // self.downscale_factor)

        batch["width"] = w
        batch["height"] = h

        # convert to float
        batch["image"] = batch["image"].float() / 255.0
        if self.apply_alpha:
            batch["alpha"] = batch["alpha"].float() / 255.0

        # Scale
        if self.downscale_factor != 1:
            batch["image"] = scale_img_nchw(batch["image"], (h, w))
            if self.apply_alpha:
                batch["segmentation"] = scale_img_nchw(batch["segmentation"].float(), (h, w), min="nearest").byte()
                batch["alpha"] = scale_img_nchw(batch["alpha"], (h, w))

        # Color correction
        if self.apply_color_correction:
            ccm = self.construct_batched_ccm(batch["camera_id"]).to(device)
            batch["image"] = (
                srgb2linear_cc(batch["image"], ccm)
                if self.color_space == "linear"
                else color_correct_srgb(batch["image"], ccm)
            )
        else:
            batch["image"] = srgb2linear(batch["image"]) if self.color_space == "linear" else batch["image"]
        batch["image"] = (
            batch["image"].sub(self.black_level_subtraction / 255.0).clamp(0, 1)
        )  # TODO: This should always be in HDR space!

        # Alpha map
        if self.apply_alpha:
            is_cloth = batch["segmentation"] == 3
            is_background = batch["segmentation"] == 0
            mask = is_cloth | is_background
            batch["alpha"] = torch.where(mask, 0.0, batch["alpha"])
            batch["image"] = batch["image"] * batch["alpha"].expand(-1, 3, -1, -1)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        seq, timestep, cam_id = self.items[idx]

        img = self.load_image(seq, timestep, cam_id)
        alpha = self.load_alpha(seq, timestep, cam_id) if self.apply_alpha else None
        seg = self.load_segmentation(seq, timestep, cam_id) if self.apply_alpha else None

        flame_params = self.load_flame_params(seq, timestep)
        flame_verts = self.load_flame_vertices(seq, timestep)

        camera_parms = self.get_camera_parameters(cam_id)
        light_info = self.load_light_info(seq, timestep)

        return {
            "image": img,
            "alpha": alpha,
            "segmentation": seg,
            "serial": cam_id,
            "frame": timestep,
            "seq": seq,
            "flame_params": flame_params,
            "flame_verts": flame_verts,
            **light_info,
            **camera_parms,
        }


def worker_init_fn(worker_id: int):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)


def collate_fn(items):
    """Modified form of `torch.utils.data.dataloader.default_collate`
    that will strip samples from the batch if they are ``None``."""
    items = [item for item in items if item is not None]
    return default_collate(items) if len(items) > 0 else None


if __name__ == "__main__":
    from becominglit.util.torchutils import to_device

    dataset = BecomingLitDataset("1001")
    assets = dataset.load_assets()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, num_workers=8, worker_init_fn=worker_init_fn, collate_fn=collate_fn
    )

    batch_filter_fn = dataset.batch_transform_fn

    for i, batch in enumerate(loader):
        batch = to_device(batch, "cuda:0")
        batch_filter_fn(batch)
        print(i)
