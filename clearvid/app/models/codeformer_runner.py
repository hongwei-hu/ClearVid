from __future__ import annotations

import importlib
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def validate_codeformer_environment(weights_path: Path | None = None) -> tuple[bool, str]:
    try:
        import torch
    except ImportError as exc:
        return False, f"CodeFormer dependencies are not installed: {exc}"

    if not torch.cuda.is_available():
        return False, "CodeFormer requires CUDA in the current runtime."

    try:
        ensure_codeformer_weights(weights_path or Path.cwd() / "weights" / "codeformer")
    except Exception as exc:  # noqa: BLE001
        return False, f"CodeFormer 权重准备失败: {exc}"

    try:
        _load_codeformer_components()
    except Exception as exc:  # noqa: BLE001
        return False, f"CodeFormer 运行时初始化失败: {exc}"

    return True, "CodeFormer environment looks available"


CODEFORMER_MODEL_FILENAME = "codeformer.pth"
CODEFORMER_MODEL_URL = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"


class CodeFormerRestorer:
    def __init__(
        self,
        fidelity_weight: float,
        upscale_factor: float,
        weights_root: Path,
        *,
        use_poisson_blend: bool = False,
    ) -> None:
        self._weights_root = weights_root
        self._fidelity_weight = fidelity_weight
        self._upscale_factor = upscale_factor
        self._use_poisson_blend = use_poisson_blend

        torch, normalize, img2tensor, tensor2img, face_restore_helper_cls, codeformer_cls = (
            _load_codeformer_components()
        )
        self._torch = torch
        self._normalize = normalize
        self._img2tensor = img2tensor
        self._tensor2img = tensor2img
        self._device = torch.device("cuda")

        codeformer_weights = ensure_codeformer_weights(weights_root / "codeformer")
        self._model = codeformer_cls(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(self._device)
        checkpoint = torch.load(str(codeformer_weights), map_location=self._device)
        state_dict = checkpoint.get("params_ema") or checkpoint.get("params") or checkpoint
        self._model.load_state_dict(state_dict, strict=True)
        self._model.eval()

        self._face_helper = face_restore_helper_cls(
            upscale_factor,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=self._device,
            model_rootpath=str(weights_root / "facelib"),
        )

    def restore_faces(self, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        self._face_helper.clean_all()
        self._face_helper.set_upscale_factor(1)
        self._face_helper.read_image(frame)

        num_faces = self._face_helper.get_face_landmarks_5(
            only_center_face=False,
            resize=640,
            eye_dist_threshold=5,
        )
        if num_faces == 0:
            return frame

        self._face_helper.align_warp_face()

        # --- Batch inference: stack all cropped faces into one tensor ---
        cropped_list = self._face_helper.cropped_faces
        batch_tensors = []
        for cropped_face in cropped_list:
            t = self._img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            self._normalize(t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            batch_tensors.append(t)

        batch = self._torch.stack(batch_tensors).to(self._device)

        try:
            with self._torch.inference_mode():
                outputs = self._model(batch, w=self._fidelity_weight, adain=True)[0]
            for i in range(outputs.shape[0]):
                restored_face = self._tensor2img(
                    outputs[i:i+1], rgb2bgr=True, min_max=(-1, 1),
                )
                self._face_helper.add_restored_face(restored_face.astype("uint8"))
            del outputs
        except Exception:  # noqa: BLE001
            # Fallback: process one-by-one if batch fails (e.g. OOM)
            for cropped_face_t in batch_tensors:
                cropped_face_t = cropped_face_t.unsqueeze(0).to(self._device)
                try:
                    with self._torch.inference_mode():
                        output = self._model(cropped_face_t, w=self._fidelity_weight, adain=True)[0]
                        restored_face = self._tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                    del output
                except Exception:  # noqa: BLE001
                    restored_face = self._tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))
                self._face_helper.add_restored_face(restored_face.astype("uint8"))

        self._face_helper.get_inverse_affine(None)
        restored_frame = self._face_helper.paste_faces_to_input_image(upsample_img=frame)

        # --- Poisson blending for smoother face boundaries ---
        if self._use_poisson_blend:
            restored_frame = _poisson_blend_faces(frame, restored_frame, self._face_helper)

        return restored_frame


def _poisson_blend_faces(
    original: np.ndarray,
    pasted: np.ndarray,
    face_helper: object,
) -> np.ndarray:
    """Apply Poisson seamless-clone blending at each detected face location.

    Falls back to *pasted* if blending fails (e.g. face near image border).
    """
    result = pasted.copy()
    h, w = result.shape[:2]

    for inv_affine in getattr(face_helper, "inverse_affine_matrices", []):
        # Estimate face center from the inverse affine matrix
        # Face region in 512x512 crop space → map center to output space
        face_center_crop = np.array([[256.0, 256.0, 1.0]])
        mapped = (inv_affine @ face_center_crop.T).T[0]
        cx, cy = int(mapped[0]), int(mapped[1])
        cx = max(1, min(cx, w - 2))
        cy = max(1, min(cy, h - 2))

        # Build an elliptical mask covering the face region
        face_size = int(256 * np.linalg.norm(inv_affine[:, 0]))
        face_size = max(20, min(face_size, min(h, w) // 2))
        mask = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.ellipse(
            mask, (cx, cy), (face_size, int(face_size * 1.3)),
            0, 0, 360, (255, 255, 255), -1,
        )

        try:
            result = cv2.seamlessClone(pasted, result, mask, (cx, cy), cv2.NORMAL_CLONE)
        except cv2.error:
            # Blending can fail if center is too close to border
            pass

    return result


def ensure_codeformer_weights(weights_path: Path) -> Path:
    weights_path.mkdir(parents=True, exist_ok=True)
    weight_file = weights_path / CODEFORMER_MODEL_FILENAME
    if weight_file.exists():
        return weight_file

    download_util = importlib.import_module("basicsr.utils.download_util")
    downloaded_path = download_util.load_file_from_url(
        url=CODEFORMER_MODEL_URL,
        model_dir=str(weights_path),
        progress=True,
        file_name=CODEFORMER_MODEL_FILENAME,
    )
    resolved = Path(downloaded_path)
    if not resolved.exists():
        raise RuntimeError(f"CodeFormer 权重下载后未找到: {resolved}")
    return resolved


def _load_codeformer_components() -> tuple[object, object, object, object, type, type]:
    torch = importlib.import_module("torch")
    transforms_functional = importlib.import_module("torchvision.transforms.functional")
    basicsr_img_util = importlib.import_module("basicsr.utils.img_util")
    face_helper_module = importlib.import_module("facexlib.utils.face_restoration_helper")
    codeformer_arch_module = importlib.import_module("basicsr.archs.codeformer_arch")
    return (
        torch,
        transforms_functional.normalize,
        basicsr_img_util.img2tensor,
        basicsr_img_util.tensor2img,
        face_helper_module.FaceRestoreHelper,
        codeformer_arch_module.CodeFormer,
    )
