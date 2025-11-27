import json
import logging
import shutil
from pathlib import Path

import numpy as np
import torch
import trimesh
import tyro
import yaml
from tqdm import tqdm

from becominglit.data.preprocess.tracking_config import TrackingConfig
from becominglit.util.env import BECOMINGLIT_DATASET_PATH, BECOMINGLIT_FLAME_TRACKING_PATH
from becominglit.util.flame import FlameHead

logger = logging.getLogger(__name__)


class TrackedFlameExporter:
    def __init__(
        self,
        subject: str,
        sequence: str,
        /,
    ):
        self.device = torch.device("cuda:0")
        self.subject = subject
        self.sequence = sequence
        self.data_root = Path(BECOMINGLIT_DATASET_PATH)
        frame_experiments_dir = Path(BECOMINGLIT_FLAME_TRACKING_PATH) / subject / sequence
        assert frame_experiments_dir.exists(), f"Sequence folder {frame_experiments_dir} does not exist"

        self.seq_folder = Path(BECOMINGLIT_DATASET_PATH) / subject / "sequences" / sequence

        self.param_folder = sorted(frame_experiments_dir.glob("20*"))[-1]  # Get the latest folder
        config_path = self.param_folder / "config.yml"
        with open(config_path, mode="r") as f:
            self.cfg: TrackingConfig = yaml.unsafe_load(f)

        # Set up FLAME model
        self.flame_model = FlameHead(
            self.cfg.model.n_shape,
            self.cfg.model.n_expr,
            add_teeth=self.cfg.model.add_teeth,
        ).to(self.device)

        # Load FLAME params
        self.flame_params = np.load(self.param_folder / "tracked_flame_params_30.npz")
        self.shape = torch.as_tensor(self.flame_params["shape"][None, ...], dtype=torch.float32, device=self.device)
        self.static_offset = (
            torch.as_tensor(self.flame_params["static_offset"], dtype=torch.float32, device=self.device)
            if "static_offset" in self.flame_params
            else torch.zeros(1, self.flame_model.v_template.shape[0], 3, dtype=torch.float32, device=self.device)
        )

        with open(self.seq_folder / "light_pattern_per_frame.json", "r") as f:
            light_pattern = json.load(f)

        self.frame_list = [lp[0] for lp in light_pattern]
        self.tracking_frames = [lp[0] for lp in light_pattern if lp[1] == 0]

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
            return tid - 2, tid + 2
        else:
            raise ValueError(f"Invalid timestep {tid}")

    def get_flame_params_for_timestep(self, tid):
        if tid < self.tracking_frames[0]:
            # Pad to first tracking frame
            params = {
                k: torch.as_tensor(v[0], dtype=torch.float32, device=self.device)
                for k, v in self.flame_params.items()
                if isinstance(v, np.ndarray)
                and k in {"expr", "jaw_pose", "neck_pose", "eyes_pose", "rotation", "translation"}
                and v.ndim > 1
            }
        elif tid > self.tracking_frames[-1]:
            # Pad to last tracking frame
            params = {
                k: torch.as_tensor(v[-1], dtype=torch.float32, device=self.device)
                for k, v in self.flame_params.items()
                if isinstance(v, np.ndarray)
                and k in {"expr", "jaw_pose", "neck_pose", "eyes_pose", "rotation", "translation"}
                and v.ndim > 1
            }
        elif tid in self.tracking_frames:
            # Tracking frame
            idx = self.tracking_frames.index(tid)
            params = {
                k: torch.as_tensor(v[idx], dtype=torch.float32, device=self.device)
                for k, v in self.flame_params.items()
                if isinstance(v, np.ndarray)
                and k in {"expr", "jaw_pose", "neck_pose", "eyes_pose", "rotation", "translation"}
                and v.ndim > 1
            }
        elif tid - 1 in self.tracking_frames:
            # Tracking frame + 1
            idx = self.tracking_frames.index(tid - 1)
            params = {
                k: torch.lerp(
                    torch.as_tensor(v[idx], dtype=torch.float32, device=self.device),
                    torch.as_tensor(v[idx + 1], dtype=torch.float32, device=self.device),
                    1 / 3,
                )
                for k, v in self.flame_params.items()
                if isinstance(v, np.ndarray)
                and k in {"expr", "jaw_pose", "neck_pose", "eyes_pose", "rotation", "translation"}
                and v.ndim > 1
            }
        elif tid - 2 in self.tracking_frames:
            # Tracking frame + 2
            idx = self.tracking_frames.index(tid - 2)
            params = {
                k: torch.lerp(
                    torch.as_tensor(v[idx], dtype=torch.float32, device=self.device),
                    torch.as_tensor(v[idx + 1], dtype=torch.float32, device=self.device),
                    2 / 3,
                )
                for k, v in self.flame_params.items()
                if isinstance(v, np.ndarray)
                and k in {"expr", "jaw_pose", "neck_pose", "eyes_pose", "rotation", "translation"}
                and v.ndim > 1
            }
        else:
            raise ValueError(f"Invalid timestep {tid}")

        return params

    def export_mesh_and_params(self, flame_params, timestep_id):
        # Forward FLAME model
        verts, verts_cano, _ = self.flame_model(
            self.shape,
            flame_params["expr"][None],
            flame_params["rotation"][None],
            flame_params["neck_pose"][None],
            flame_params["jaw_pose"][None],
            flame_params["eyes_pose"][None],
            flame_params["translation"][None],
            static_offset=self.static_offset,
            return_verts_cano=True,
            # zero_centered_at_root_node=True,
        )

        # Posed mesh export
        posed_mesh_export_path = self.data_root.joinpath(
            self.subject, "sequences", self.sequence, "flame_tracking", "meshes_posed", f"frame_{timestep_id:06d}.ply"
        )
        posed_mesh_export_path.parent.mkdir(parents=True, exist_ok=True)
        trimesh.Trimesh(vertices=verts[0].detach().cpu().numpy(), faces=self.flame_model.faces.cpu().numpy()).export(
            posed_mesh_export_path
        )

        # Canonical mesh export
        cano_mesh_export_path = self.data_root.joinpath(
            self.subject, "sequences", self.sequence, "flame_tracking", "meshes_cano", f"frame_{timestep_id:06d}.ply"
        )
        cano_mesh_export_path.parent.mkdir(parents=True, exist_ok=True)
        trimesh.Trimesh(
            vertices=verts_cano[0].detach().cpu().numpy(), faces=self.flame_model.faces.cpu().numpy()
        ).export(cano_mesh_export_path)

        # Export FLAME parameters
        flame_param_path = self.data_root.joinpath(
            self.subject, "sequences", self.sequence, "flame_tracking", "flame_params", f"frame_{timestep_id:06d}.npz"
        )
        flame_param_path.parent.mkdir(exist_ok=True, parents=True)
        np.savez(
            flame_param_path,
            shape=self.shape.detach().cpu().numpy(),
            expr=flame_params["expr"].detach().cpu().numpy(),
            jaw_pose=flame_params["jaw_pose"].detach().cpu().numpy(),
            neck_pose=flame_params["neck_pose"].detach().cpu().numpy(),
            eyes_pose=flame_params["eyes_pose"].detach().cpu().numpy(),
            rotation=flame_params["rotation"].detach().cpu().numpy(),
            translation=flame_params["translation"].detach().cpu().numpy(),
            static_offset=self.static_offset.detach().cpu().numpy(),
        )

    def export(self):
        # export albedo texture
        albedo_path = self.param_folder.joinpath("tracked_flame_params_30_albedo.png")
        if albedo_path.exists():
            logger.info("Exporting albedo texture to assets folder")
            albedo_export_path = self.data_root.joinpath(self.subject, "assets", "color_mean.png")
            albedo_export_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(albedo_path, albedo_export_path)

        for tid in tqdm(
            self.frame_list,
            desc=f"Exporting FLAME params for SID {self.subject}, SEQ {self.sequence}",
        ):
            flame_params = self.get_flame_params_for_timestep(tid)
            self.export_mesh_and_params(flame_params, tid)

            torch.cuda.empty_cache()


if __name__ == "__main__":
    exporter = tyro.cli(TrackedFlameExporter)
    exporter.export()
