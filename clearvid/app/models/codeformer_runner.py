from __future__ import annotations

import importlib
from pathlib import Path

import cv2


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
    ) -> None:
        self._weights_root = weights_root
        self._fidelity_weight = fidelity_weight
        self._upscale_factor = upscale_factor

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
        for cropped_face in self._face_helper.cropped_faces:
            cropped_face_t = self._img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            self._normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self._device)

            try:
                with self._torch.no_grad():
                    output = self._model(cropped_face_t, w=self._fidelity_weight, adain=True)[0]
                    restored_face = self._tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
            except Exception:  # noqa: BLE001
                restored_face = self._tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

            self._face_helper.add_restored_face(restored_face.astype("uint8"))

        self._face_helper.get_inverse_affine(None)
        restored_frame = self._face_helper.paste_faces_to_input_image(upsample_img=frame)
        self._torch.cuda.empty_cache()
        return restored_frame


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
