"""Right sidebar: export settings panel with collapsible sections and tooltips."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from clearvid.app.bootstrap.paths import OUTPUTS_DIR

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from clearvid.app.gui._helpers import coerce_enum, populate_combo, set_combo_by_value
from clearvid.app.gui.estimation import estimate_export, format_duration
from clearvid.app.gui.naming import DEFAULT_TEMPLATE, render_output_name
from clearvid.app.gui.preset_cards import BUILTIN_PRESETS, Preset, PresetCardsWidget
from clearvid.app.gui.widgets.collapsible import CollapsibleSection
from clearvid.app.gui.widgets.hint_label import labeled_row_with_info
from clearvid.app.schemas.models import (
    BackendType,
    EnhancementConfig,
    FaceRestoreModel,
    InferenceAccelerator,
    QualityMode,
    TargetProfile,
    UpscaleModel,
)

# ---------------------------------------------------------------------------
# Label dictionaries
# ---------------------------------------------------------------------------

BACKEND_LABELS = {
    BackendType.AUTO: "自动",
    BackendType.BASELINE: "基线增强",
    BackendType.REALESRGAN: "Real-ESRGAN",
}
UPSCALE_MODEL_LABELS = {
    UpscaleModel.AUTO: "自动（质量模式决定）",
    UpscaleModel.GENERAL_V3: "General v3 （轻量快速）",
    UpscaleModel.X4PLUS: "x4plus RRDB （高质量）",
}
ACCELERATOR_LABELS = {
    InferenceAccelerator.AUTO: "自动检测",
    InferenceAccelerator.NONE: "无加速",
    InferenceAccelerator.COMPILE: "torch.compile",
    InferenceAccelerator.TENSORRT: "TensorRT",
}
FACE_MODEL_LABELS = {
    FaceRestoreModel.CODEFORMER: "CodeFormer",
    FaceRestoreModel.GFPGAN: "GFPGAN",
}
QUALITY_LABELS = {
    QualityMode.FAST: "快速（无后处理）",
    QualityMode.BALANCED: "平衡",
    QualityMode.QUALITY: "高质量（慢 6x）",
}
TARGET_LABELS = {
    TargetProfile.SOURCE: "保持原始分辨率",
    TargetProfile.FHD: "1080p",
    TargetProfile.UHD4K: "4K",
    TargetProfile.SCALE2X: "放大 2 倍",
    TargetProfile.SCALE4X: "放大 4 倍",
}

# ---------------------------------------------------------------------------
# Tooltip texts (user-friendly explanations for every parameter)
# ---------------------------------------------------------------------------

TOOLTIPS: dict[str, str] = {
    "target_profile": (
        "目标输出分辨率。\n"
        "480p 视频建议选 1080p，720p/1080p 视频可选 4K。\n"
        "选「保持原始」则只做画质增强不改变大小"
    ),
    "quality_mode": (
        "控制 AI 超分精度和后处理流程。\n"
        "快速: 轻量模型 + 跳过人脸修复和时序稳定，速度最快\n"
        "平衡: 轻量模型 + 完整后处理\n"
        "高质量: 重型模型 + 完整后处理，效果最好但约慢 6 倍"
    ),
    "backend": (
        "Real-ESRGAN 是核心 AI 超分引擎，需要 NVIDIA 独显。\n"
        "基线模式仅使用传统滤镜，质量较低但无需 GPU"
    ),
    "upscale_model": (
        "General v3 轻量快速，适合批量处理；\n"
        "x4plus 参数量更大细节更丰富，适合精品制作"
    ),
    "encoder_crf": (
        "恒定质量系数，数字越小画质越好但文件越大。\n"
        "推荐: 15=近无损  18=高质量  22=标准  28=较低"
    ),
    "pixel_format": (
        "10-bit 色彩过渡更平滑（减少色带），\n"
        "但部分旧设备可能不支持播放"
    ),
    "face_restore": (
        "AI 自动检测并修复模糊人脸。\n"
        "没有人物或仅有远景时关闭可加快处理速度"
    ),
    "face_strength": (
        "低 = 保守修复，保留更多原始细节\n"
        "高 = 激进修复，人脸更清晰但可能不够自然\n"
        "建议范围: 0.4 - 0.7"
    ),
    "face_model": "CodeFormer 更自然保真；GFPGAN 偏向美化肤质",
    "poisson_blend": "让修复后的人脸与周围皮肤过渡更自然，减少色差和接缝感",
    "temporal": "AI 超分逐帧独立处理可能导致纹理闪烁，开启后利用相邻帧信息抑制闪烁",
    "temporal_strength": "越高越稳定但可能损失运动细节。快速运动视频建议调低",
    "sharpen": "增强画面边缘清晰度。过高可能产生不自然的白边，建议 0.05-0.20",
    "denoise": "自动根据视频码率调整降噪力度。低码率视频强降噪，高码率轻降噪以保留细节",
    "deblock": "低码率 H.264 视频常见方块状伪影，此选项专门修复。高码率视频会自动跳过",
    "deinterlace": "自动检测隔行扫描(老旧录像常见)，转为逐行扫描消除运动时的横条纹",
    "colorspace": "不同来源视频可能使用不同色彩标准(BT.601/709)，统一后 AI 模型处理效果更一致",
    "accelerator": "TensorRT 可将处理速度提升 2-4 倍，但首次使用需要编译引擎(约 3-10 分钟)",
    "async_pipeline": "让解码→AI增强→编码三步同时运行，可提速约 30-50%",
    "batch_size": "每次送入 GPU 的帧数。0 = 根据显存自动选择。\n显存大可以调高(8-16)以提升吞吐量",
    "tile_size": "分块尺寸。0 = 自动(整帧处理或根据显存分块)。\n显存不足时可设 256/512 降低峰值显存占用",
    "preserve_audio": "保留原视频的音频轨道。关闭后导出为纯视频（无声）",
    "preserve_subtitles": "保留嵌入的字幕轨道（如有）",
    "preserve_metadata": "保留拍摄日期、GPS 等元信息",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _hint(text: str) -> QLabel:
    """Create a small gray hint label below a control."""
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet("color: #9e9e9e; font-size: 11px; margin-left: 4px;")
    return lbl


def _labeled_row(
    label_text: str, widget: QWidget, tooltip: str = ""
) -> QHBoxLayout:
    """Create ``Label | Widget | ℹ️`` horizontal row with optional tooltip and info button."""
    return labeled_row_with_info(label_text, widget, tooltip=tooltip, detail=tooltip)


# ===========================================================================
# ExportPanel
# ===========================================================================


class ExportPanel(QWidget):
    """Right sidebar containing all export parameters in collapsible sections."""

    export_requested = Signal()
    export_all_requested = Signal()  # queue all files in file list
    smart_params_requested = Signal()
    output_dir_changed = Signal(str)
    pause_requested = Signal()
    cancel_requested = Signal()
    log_message = Signal(str)  # forwards messages to the main window log panel

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sections: dict[str, CollapsibleSection] = {}
        self._last_estimate: object | None = None

        # Outer layout: scrollable body + fixed bottom
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(4)

        self._build_smart_section()
        self._build_preset_section()
        self._build_output_section()
        self._build_encoding_section()
        self._build_enhancement_section()
        self._build_face_section()
        self._build_preprocess_section()
        self._build_temporal_section()
        self._build_stream_section()
        self._build_performance_section()

        self._layout.addStretch(1)

        scroll.setWidget(container)
        outer.addWidget(scroll, 1)

        # --- Fixed bottom: progress + export button ---
        self._build_bottom(outer)

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_smart_section(self) -> None:
        self._smart_btn = QPushButton("\u26a1 一键最佳适配")
        self._smart_btn.setObjectName("smartButton")
        self._smart_btn.setMinimumHeight(34)
        self._smart_btn.setToolTip("根据输入视频和硬件自动选择最佳处理参数")
        self._smart_btn.clicked.connect(self.smart_params_requested.emit)
        self._layout.addWidget(self._smart_btn)

    def _build_preset_section(self) -> None:
        label = QLabel("快速预设")
        label.setStyleSheet(
            "color: #4fc3f7; font-weight: bold; font-size: 12px; margin-top: 6px;"
        )
        self._layout.addWidget(label)

        self._preset_cards = PresetCardsWidget()
        self._preset_cards.preset_selected.connect(self._on_preset_selected)
        self._layout.addWidget(self._preset_cards)

    def _build_output_section(self) -> None:
        sec = CollapsibleSection("输出设置", name="output", expanded=True)
        lay = sec.content_layout

        self.target_combo = QComboBox()
        populate_combo(self.target_combo, TARGET_LABELS, TargetProfile, TargetProfile.FHD)
        lay.addLayout(_labeled_row("输出规格", self.target_combo, TOOLTIPS["target_profile"]))
        lay.addWidget(_hint("目标分辨率，480p\u21921080p / 720p\u21924K"))

        # Naming template
        self.naming_edit = QLineEdit()
        self.naming_edit.setText(DEFAULT_TEMPLATE)
        self.naming_edit.setPlaceholderText("{name}_{profile}")
        lay.addLayout(_labeled_row(
            "命名规则", self.naming_edit,
            "支持变量: {name}=原文件名, {profile}=输出规格, {date}=日期, {time}=时间",
        ))
        lay.addWidget(_hint("{name}_{profile} → 视频名_fhd.mp4"))

        # Export duration (preview_seconds)
        self.preview_seconds = QSpinBox()
        self.preview_seconds.setMinimum(0)
        self.preview_seconds.setMaximum(24 * 60 * 60)
        self.preview_seconds.setValue(0)
        self.preview_seconds.setSuffix(" 秒")
        self.preview_seconds.setSpecialValueText("完整视频")
        self.preview_seconds.valueChanged.connect(self._on_export_duration_changed)
        lay.addLayout(_labeled_row(
            "导出时长", self.preview_seconds,
            "仅处理视频前 N 秒，设为 0 则导出全片。适合快速试效果。",
        ))
        lay.addWidget(_hint("0 = 全片；输入秒数可只导出前段"))

        # Output path
        out_row = QHBoxLayout()
        out_lbl = QLabel("输出路径")
        out_lbl.setMinimumWidth(80)
        out_row.addWidget(out_lbl)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("选择输出位置...")
        out_row.addWidget(self.output_edit, 1)
        out_browse = QPushButton("\U0001f4c1")
        out_browse.setFixedWidth(32)
        out_browse.setToolTip("浏览输出位置")
        out_browse.clicked.connect(self._browse_output)
        out_row.addWidget(out_browse)
        lay.addLayout(out_row)

        self._layout.addWidget(sec)
        self._sections["output"] = sec

    def _build_encoding_section(self) -> None:
        sec = CollapsibleSection("编码设置", name="encoding", expanded=True)
        lay = sec.content_layout

        self.encoder_crf = QSpinBox()
        self.encoder_crf.setMinimum(0)
        self.encoder_crf.setMaximum(63)
        self.encoder_crf.setValue(18)
        self.encoder_crf.setSpecialValueText("自动")
        lay.addLayout(_labeled_row("画质 CRF", self.encoder_crf, TOOLTIPS["encoder_crf"]))
        lay.addWidget(_hint("越小画质越好，推荐 15-22"))

        self.pixel_format_combo = QComboBox()
        self.pixel_format_combo.addItem("8-bit (yuv420p)", "yuv420p")
        self.pixel_format_combo.addItem("10-bit (yuv420p10le)", "yuv420p10le")
        self.pixel_format_combo.addItem("10-bit (p010le)", "p010le")
        lay.addLayout(
            _labeled_row("色彩深度", self.pixel_format_combo, TOOLTIPS["pixel_format"])
        )

        self._layout.addWidget(sec)
        self._sections["encoding"] = sec

    def _build_enhancement_section(self) -> None:
        sec = CollapsibleSection("画质增强", name="enhancement", expanded=False)
        lay = sec.content_layout

        self.backend_combo = QComboBox()
        populate_combo(self.backend_combo, BACKEND_LABELS, BackendType, BackendType.AUTO)
        lay.addLayout(_labeled_row("增强后端", self.backend_combo, TOOLTIPS["backend"]))
        lay.addWidget(_hint("自动根据显卡选择最佳方案"))

        self.upscale_model_combo = QComboBox()
        populate_combo(
            self.upscale_model_combo, UPSCALE_MODEL_LABELS, UpscaleModel, UpscaleModel.AUTO
        )
        lay.addLayout(
            _labeled_row("超分模型", self.upscale_model_combo, TOOLTIPS["upscale_model"])
        )

        self.quality_combo = QComboBox()
        populate_combo(self.quality_combo, QUALITY_LABELS, QualityMode, QualityMode.QUALITY)
        lay.addLayout(_labeled_row("质量模式", self.quality_combo, TOOLTIPS["quality_mode"]))

        self.sharpen_enabled = QCheckBox("启用锐化")
        self.sharpen_enabled.setChecked(True)
        self.sharpen_enabled.setToolTip(TOOLTIPS["sharpen"])
        lay.addWidget(self.sharpen_enabled)

        self.sharpen_strength = QDoubleSpinBox()
        self.sharpen_strength.setDecimals(2)
        self.sharpen_strength.setRange(0.0, 1.0)
        self.sharpen_strength.setSingleStep(0.05)
        self.sharpen_strength.setValue(0.12)
        lay.addLayout(_labeled_row("锐化强度", self.sharpen_strength, TOOLTIPS["sharpen"]))
        lay.addWidget(_hint("建议 0.05-0.20，过高会产生白边"))

        self._layout.addWidget(sec)
        self._sections["enhancement"] = sec

    def _build_face_section(self) -> None:
        sec = CollapsibleSection("人脸修复", name="face", expanded=False)
        lay = sec.content_layout

        self.face_restore_enabled = QCheckBox("启用人脸修复")
        self.face_restore_enabled.setChecked(False)
        self.face_restore_enabled.setToolTip(TOOLTIPS["face_restore"])
        lay.addWidget(self.face_restore_enabled)
        lay.addWidget(_hint("自动检测并修复模糊人脸"))

        self.face_restore_strength = QDoubleSpinBox()
        self.face_restore_strength.setDecimals(2)
        self.face_restore_strength.setRange(0.0, 1.0)
        self.face_restore_strength.setSingleStep(0.05)
        self.face_restore_strength.setValue(0.55)
        lay.addLayout(
            _labeled_row("修复强度", self.face_restore_strength, TOOLTIPS["face_strength"])
        )
        lay.addWidget(_hint("0.4-0.7 较自然，过高可能不真实"))

        self.face_model_combo = QComboBox()
        populate_combo(
            self.face_model_combo, FACE_MODEL_LABELS, FaceRestoreModel, FaceRestoreModel.CODEFORMER
        )
        lay.addLayout(_labeled_row("修复模型", self.face_model_combo, TOOLTIPS["face_model"]))

        self.face_poisson_blend = QCheckBox("泊松融合")
        self.face_poisson_blend.setChecked(False)
        self.face_poisson_blend.setToolTip(TOOLTIPS["poisson_blend"])
        lay.addWidget(self.face_poisson_blend)
        lay.addWidget(_hint("让修复后的人脸边缘过渡更自然"))

        self._layout.addWidget(sec)
        self._sections["face"] = sec

    def _build_preprocess_section(self) -> None:
        sec = CollapsibleSection("预处理", name="preprocess", expanded=False)
        lay = sec.content_layout

        self.preprocess_denoise = QCheckBox("智能降噪")
        self.preprocess_denoise.setChecked(False)  # default off; auto-enabled by recommend()
        self.preprocess_denoise.setToolTip(TOOLTIPS["denoise"])
        lay.addWidget(self.preprocess_denoise)
        lay.addWidget(_hint("去除噪点颗粒，对低码率视频效果明显（会降低解码吞吐，建议由『一键最佳』自动判断）"))

        self.preprocess_deblock = QCheckBox("去块效应")
        self.preprocess_deblock.setChecked(True)
        self.preprocess_deblock.setToolTip(TOOLTIPS["deblock"])
        lay.addWidget(self.preprocess_deblock)
        lay.addWidget(_hint("修复低码率视频的方块感"))

        self.preprocess_deinterlace = QCheckBox("自动去隔行")
        self.preprocess_deinterlace.setChecked(True)
        self.preprocess_deinterlace.setToolTip(TOOLTIPS["deinterlace"])
        lay.addWidget(self.preprocess_deinterlace)

        self.preprocess_colorspace = QCheckBox("色彩空间归一化")
        self.preprocess_colorspace.setChecked(True)
        self.preprocess_colorspace.setToolTip(TOOLTIPS["colorspace"])
        lay.addWidget(self.preprocess_colorspace)

        self._layout.addWidget(sec)
        self._sections["preprocess"] = sec

    def _build_temporal_section(self) -> None:
        sec = CollapsibleSection("时序与稳定", name="temporal", expanded=False)
        lay = sec.content_layout

        self.temporal_enabled = QCheckBox("启用时序稳定")
        self.temporal_enabled.setChecked(True)
        self.temporal_enabled.setToolTip(TOOLTIPS["temporal"])
        lay.addWidget(self.temporal_enabled)
        lay.addWidget(_hint("减少 AI 超分后相邻帧的闪烁"))

        self.temporal_strength = QDoubleSpinBox()
        self.temporal_strength.setDecimals(2)
        self.temporal_strength.setRange(0.0, 1.0)
        self.temporal_strength.setSingleStep(0.05)
        self.temporal_strength.setValue(0.6)
        lay.addLayout(
            _labeled_row("稳定强度", self.temporal_strength, TOOLTIPS["temporal_strength"])
        )
        lay.addWidget(_hint("快速运动视频建议调低"))

        self._layout.addWidget(sec)
        self._sections["temporal"] = sec

    def _build_stream_section(self) -> None:
        sec = CollapsibleSection("流保留", name="stream", expanded=False)
        lay = sec.content_layout

        self.preserve_audio = QCheckBox("保留音频")
        self.preserve_audio.setChecked(True)
        self.preserve_audio.setToolTip(TOOLTIPS["preserve_audio"])
        lay.addWidget(self.preserve_audio)

        self.preserve_subtitles = QCheckBox("保留字幕")
        self.preserve_subtitles.setChecked(True)
        self.preserve_subtitles.setToolTip(TOOLTIPS["preserve_subtitles"])
        lay.addWidget(self.preserve_subtitles)

        self.preserve_metadata = QCheckBox("保留元数据")
        self.preserve_metadata.setChecked(True)
        self.preserve_metadata.setToolTip(TOOLTIPS["preserve_metadata"])
        lay.addWidget(self.preserve_metadata)

        self._layout.addWidget(sec)
        self._sections["stream"] = sec

    def _build_performance_section(self) -> None:
        sec = CollapsibleSection("性能", name="performance", expanded=False)
        lay = sec.content_layout

        self.accelerator_combo = QComboBox()
        populate_combo(
            self.accelerator_combo,
            ACCELERATOR_LABELS,
            InferenceAccelerator,
            InferenceAccelerator.AUTO,
        )
        lay.addLayout(
            _labeled_row("推理加速", self.accelerator_combo, TOOLTIPS["accelerator"])
        )

        # --- TensorRT deploy row ---
        trt_deploy_row = QHBoxLayout()
        self.trt_status_label = QLabel("")
        self.trt_status_label.setStyleSheet(
            "font-size: 11px; color: #888; padding-left: 4px;"
        )
        self.trt_deploy_btn = QPushButton("部署 TensorRT 引擎")
        self.trt_deploy_btn.setFixedWidth(160)
        self.trt_deploy_btn.setVisible(False)
        self.trt_deploy_progress = QProgressBar()
        self.trt_deploy_progress.setRange(0, 100)
        self.trt_deploy_progress.setValue(0)
        self.trt_deploy_progress.setVisible(False)
        self.trt_deploy_progress.setFixedWidth(160)
        self.trt_deploy_progress.setFixedHeight(18)
        self.trt_deploy_progress.setTextVisible(True)
        self.trt_deploy_progress.setFormat("")
        # Toggling visibility doesn't change the row height — we use a
        # QStackedWidget-style approach: show either button or bar.
        trt_deploy_row.addWidget(self.trt_deploy_btn)
        trt_deploy_row.addWidget(self.trt_deploy_progress)
        trt_deploy_row.addWidget(self.trt_status_label)
        trt_deploy_row.addStretch()
        lay.addLayout(trt_deploy_row)

        self._trt_deploying = False
        self._trt_warmup_worker = None
        self._is_exporting = False  # set by MainWindow via set_exporting_state()

        self.trt_deploy_btn.clicked.connect(self._start_trt_deploy)

        # Connect accelerator combo to show/hide and re-check deploy state
        self.accelerator_combo.currentIndexChanged.connect(
            self._on_accelerator_changed,
        )

        self.async_pipeline = QCheckBox("异步流水线")
        self.async_pipeline.setChecked(True)
        self.async_pipeline.setToolTip(TOOLTIPS["async_pipeline"])
        lay.addWidget(self.async_pipeline)
        lay.addWidget(_hint("三级并行处理，提速约 30-50%"))

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(0, 64)
        self.batch_size_spin.setValue(0)
        self.batch_size_spin.setSpecialValueText("自动")
        lay.addLayout(
            _labeled_row("批处理帧数", self.batch_size_spin, TOOLTIPS["batch_size"])
        )
        lay.addWidget(_hint("0=自动，显存≥24GB 可尝试 8-16"))

        self.tile_size_spin = QSpinBox()
        self.tile_size_spin.setRange(0, 2048)
        self.tile_size_spin.setSingleStep(128)
        self.tile_size_spin.setValue(0)
        self.tile_size_spin.setSpecialValueText("自动")
        lay.addLayout(
            _labeled_row("分块尺寸", self.tile_size_spin, TOOLTIPS["tile_size"])
        )
        lay.addWidget(_hint("0=自动，OOM 时可设 256/512"))

        self._layout.addWidget(sec)
        self._sections["performance"] = sec

    def _build_bottom(self, outer: QVBoxLayout) -> None:
        """Build the fixed bottom area: estimation, progress bar, export buttons."""
        bottom = QVBoxLayout()
        bottom.setContentsMargins(8, 8, 8, 8)
        bottom.setSpacing(6)

        # Estimation label
        self.estimation_label = QLabel("")
        self.estimation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.estimation_label.setStyleSheet(
            "color: #81c784; font-size: 11px; padding: 2px;"
        )
        self.estimation_label.setVisible(False)
        bottom.addWidget(self.estimation_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        bottom.addWidget(self.progress_bar)

        self.progress_label = QLabel("就绪")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setStyleSheet("color: #9e9e9e; font-size: 12px;")
        bottom.addWidget(self.progress_label)

        self.export_btn = QPushButton("\u25b6  开始导出")
        self.export_btn.setObjectName("exportButton")
        self.export_btn.setMinimumHeight(40)
        self.export_btn.clicked.connect(self.export_requested.emit)
        bottom.addWidget(self.export_btn)

        # Queue export button
        self.export_all_btn = QPushButton("📋  全部导出")
        self.export_all_btn.setMinimumHeight(32)
        self.export_all_btn.setToolTip("将文件列表中的所有视频排入队列依次处理")
        self.export_all_btn.clicked.connect(self.export_all_requested.emit)
        bottom.addWidget(self.export_all_btn)

        # Pause / Cancel buttons (visible only during export)
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)
        self._pause_btn = QPushButton("⏸ 暂停")
        self._pause_btn.setMinimumHeight(32)
        self._pause_btn.setToolTip("暂停/继续当前导出")
        self._pause_btn.setVisible(False)
        self._pause_btn.clicked.connect(self._toggle_pause)
        ctrl_row.addWidget(self._pause_btn)

        self._cancel_btn = QPushButton("⏹ 取消")
        self._cancel_btn.setMinimumHeight(32)
        self._cancel_btn.setToolTip("取消当前导出 (Esc)")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self.cancel_requested.emit)
        ctrl_row.addWidget(self._cancel_btn)
        bottom.addLayout(ctrl_row)

        self._is_paused = False

        # Mid-export preview button (hidden until preview is available)
        self._preview_progress_btn = QPushButton("👁 预览已完成部分")
        self._preview_progress_btn.setMinimumHeight(32)
        self._preview_progress_btn.setToolTip(
            "在系统播放器中播放已导出的视频片段（含音频）"
        )
        self._preview_progress_btn.setVisible(False)
        self._preview_progress_path: str = ""
        self._preview_progress_btn.clicked.connect(self._play_preview_progress)
        bottom.addWidget(self._preview_progress_btn)

        # Post-export action buttons (hidden until export completes)
        self._post_row = QHBoxLayout()
        self._open_folder_btn = QPushButton("\U0001f4c2 打开文件夹")
        self._open_folder_btn.setVisible(False)
        self._play_btn = QPushButton("\u25b6 播放视频")
        self._play_btn.setVisible(False)
        self._post_row.addWidget(self._open_folder_btn)
        self._post_row.addWidget(self._play_btn)
        bottom.addLayout(self._post_row)

        outer.addLayout(bottom)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_sections(self) -> dict[str, CollapsibleSection]:
        return self._sections

    def _on_preset_selected(self, preset: Preset) -> None:
        """Apply a preset's parameter dict to the widget state."""
        p = preset.params
        if not p:
            return  # "自定义" — do nothing

        _COMBO_MAP = {
            "quality_mode": (self.quality_combo, QualityMode),
            "upscale_model": (self.upscale_model_combo, UpscaleModel),
            "face_restore_model": (self.face_model_combo, FaceRestoreModel),
        }
        for key, (combo, _enum) in _COMBO_MAP.items():
            if key in p:
                set_combo_by_value(combo, p[key])

        _CHECK_MAP = {
            "face_restore_enabled": self.face_restore_enabled,
            "face_poisson_blend": self.face_poisson_blend,
            "sharpen_enabled": self.sharpen_enabled,
            "temporal_stabilize_enabled": self.temporal_enabled,
            "preprocess_denoise": self.preprocess_denoise,
            "preprocess_deblock": self.preprocess_deblock,
            "preprocess_deinterlace": self.preprocess_deinterlace,
            "preprocess_colorspace": self.preprocess_colorspace,
        }
        for key, checkbox in _CHECK_MAP.items():
            if key in p:
                checkbox.setChecked(p[key])

        _SPIN_MAP = {
            "face_restore_strength": self.face_restore_strength,
            "sharpen_strength": self.sharpen_strength,
            "temporal_stabilize_strength": self.temporal_strength,
        }
        for key, spinbox in _SPIN_MAP.items():
            if key in p:
                spinbox.setValue(p[key])

        if "encoder_crf" in p:
            self.encoder_crf.setValue(p["encoder_crf"])

    def build_config(self, input_path: str) -> EnhancementConfig:
        """Build an EnhancementConfig from the current widget state."""
        return EnhancementConfig(
            input_path=Path(input_path),
            output_path=Path(self.output_edit.text()),
            target_profile=coerce_enum(
                TargetProfile, self.target_combo.currentData(), TargetProfile.FHD
            ),
            quality_mode=coerce_enum(
                QualityMode, self.quality_combo.currentData(), QualityMode.QUALITY
            ),
            backend=coerce_enum(
                BackendType, self.backend_combo.currentData(), BackendType.AUTO
            ),
            upscale_model=coerce_enum(
                UpscaleModel, self.upscale_model_combo.currentData(), UpscaleModel.AUTO
            ),
            inference_accelerator=coerce_enum(
                InferenceAccelerator,
                self.accelerator_combo.currentData(),
                InferenceAccelerator.AUTO,
            ),
            async_pipeline=self.async_pipeline.isChecked(),
            batch_size=self.batch_size_spin.value(),
            tile_size=self.tile_size_spin.value(),
            face_restore_enabled=self.face_restore_enabled.isChecked(),
            face_restore_strength=self.face_restore_strength.value(),
            face_restore_model=coerce_enum(
                FaceRestoreModel,
                self.face_model_combo.currentData(),
                FaceRestoreModel.CODEFORMER,
            ),
            face_poisson_blend=self.face_poisson_blend.isChecked(),
            sharpen_enabled=self.sharpen_enabled.isChecked(),
            sharpen_strength=self.sharpen_strength.value(),
            encoder_crf=self.encoder_crf.value() if self.encoder_crf.value() > 0 else None,
            output_pixel_format=self.pixel_format_combo.currentData() or "yuv420p",
            temporal_stabilize_enabled=self.temporal_enabled.isChecked(),
            temporal_stabilize_strength=self.temporal_strength.value(),
            preprocess_denoise=self.preprocess_denoise.isChecked(),
            preprocess_deblock=self.preprocess_deblock.isChecked(),
            preprocess_deinterlace=(
                "auto" if self.preprocess_deinterlace.isChecked() else "off"
            ),
            preprocess_colorspace_normalize=self.preprocess_colorspace.isChecked(),
            preserve_audio=self.preserve_audio.isChecked(),
            preserve_subtitles=self.preserve_subtitles.isChecked(),
            preserve_metadata=self.preserve_metadata.isChecked(),
            preview_seconds=self.preview_seconds.value() or None,
        )

    def build_preview_config(self, input_path: str) -> EnhancementConfig:
        """Build a minimal config for preview (no real output needed)."""
        return EnhancementConfig(
            input_path=Path(input_path),
            output_path=Path(self.output_edit.text() or "preview_temp.mp4"),
            target_profile=coerce_enum(
                TargetProfile, self.target_combo.currentData(), TargetProfile.FHD
            ),
            quality_mode=coerce_enum(
                QualityMode, self.quality_combo.currentData(), QualityMode.QUALITY
            ),
            upscale_model=coerce_enum(
                UpscaleModel, self.upscale_model_combo.currentData(), UpscaleModel.AUTO
            ),
            face_restore_enabled=self.face_restore_enabled.isChecked(),
            face_restore_strength=self.face_restore_strength.value(),
            face_restore_model=coerce_enum(
                FaceRestoreModel,
                self.face_model_combo.currentData(),
                FaceRestoreModel.CODEFORMER,
            ),
            face_poisson_blend=self.face_poisson_blend.isChecked(),
            sharpen_enabled=self.sharpen_enabled.isChecked(),
            sharpen_strength=self.sharpen_strength.value(),
        )

    # --- TensorRT deploy -------------------------------------------------------

    def _on_accelerator_changed(self) -> None:
        """Show/hide the TRT deploy button based on selected accelerator."""
        accel = coerce_enum(
            InferenceAccelerator,
            self.accelerator_combo.currentData(),
            InferenceAccelerator.AUTO,
        )
        is_trt = accel == InferenceAccelerator.TENSORRT
        self.trt_deploy_btn.setVisible(is_trt and not self._trt_deploying)
        self._refresh_trt_status()

    def _start_trt_deploy(self) -> None:
        """Launch background TensorRT engine deployment."""
        from clearvid.app.gui.workers import TrtWarmupWorker
        from PySide6.QtWidgets import QMessageBox

        # Guard: refuse if export is in progress
        if getattr(self, "_is_exporting", False):
            QMessageBox.warning(
                self,
                "正在导出中",
                "当前正在进行视频导出，GPU 显存已被占用。\n"
                "请等待导出完成后再部署 TensorRT 引擎，\n"
                "否则两个任务同时抢占显存会导致 OOM 崩溃。",
            )
            return

        # Guard: VRAM check — warn if less than 6 GB free
        try:
            import torch
            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                free_mb = free_bytes // (1024 * 1024)
                total_mb = total_bytes // (1024 * 1024)
                if free_mb < 6144:
                    reply = QMessageBox.warning(
                        self,
                        "可用显存不足",
                        f"GPU 当前剩余显存仅 {free_mb / 1024:.1f} GB / {total_mb / 1024:.0f} GB。\n"
                        "TensorRT 引擎构建通常需要 6 GB 以上空闲显存。\n\n"
                        "建议先关闭其他占用 GPU 的程序后再部署。\n\n继续强制部署？",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No,
                    )
                    if reply != QMessageBox.StandardButton.Yes:
                        return
        except Exception:  # noqa: BLE001
            pass

        # Gather current config parameters
        model_key = coerce_enum(
            UpscaleModel, self.upscale_model_combo.currentData(), UpscaleModel.AUTO,
        )
        model_key_str = "general_v3" if model_key == UpscaleModel.AUTO else model_key.value
        tile = self.tile_size_spin.value() or 512
        batch = self._recommended_trt_batch()

        self._trt_deploying = True
        self.trt_deploy_btn.setVisible(False)
        self.trt_deploy_progress.setVisible(True)
        self.trt_deploy_progress.setValue(0)
        self.trt_status_label.setText("准备部署...")
        self.trt_status_label.setStyleSheet("font-size: 11px; color: #ccc; padding-left: 4px;")

        self.log_message.emit(
            f"[TRT] 开始部署: model={model_key_str}, tile={tile}, batch={batch}, fp16=True"
        )

        self._trt_warmup_worker = TrtWarmupWorker(
            model_key=model_key_str,
            tile_size=tile,
            batch_size=batch,
            low_load=False,
        )
        self._trt_warmup_worker.progress.connect(self._on_trt_deploy_progress)
        self._trt_warmup_worker.failed.connect(self._on_trt_deploy_failed)
        self._trt_warmup_worker.ready.connect(lambda _: None)  # noop
        self._trt_warmup_worker.done.connect(self._on_trt_deploy_done)
        self._trt_warmup_worker.start()

    def _on_trt_deploy_progress(self, pct: int, msg: str) -> None:
        self.trt_deploy_progress.setValue(pct)
        self.trt_status_label.setText(msg)
        self.log_message.emit(f"[TRT {pct:3d}%] {msg}")
        if pct >= 100:
            self.trt_status_label.setStyleSheet(
                "font-size: 11px; color: #4caf50; padding-left: 4px;"
            )

    def _on_trt_deploy_failed(self, err: str) -> None:
        short = err.split("\n")[0]  # first line is the core message
        self.trt_status_label.setText(f"部署失败: {short[:120]}")
        self.trt_status_label.setStyleSheet(
            "font-size: 11px; color: #f44336; padding-left: 4px;"
        )
        self.log_message.emit(f"[TRT 失败]\n{err}")

    def _on_trt_deploy_done(self) -> None:
        self._trt_deploying = False
        self.trt_deploy_progress.setVisible(False)
        self.log_message.emit("[TRT] 部署流程结束")
        self._refresh_trt_status()

    def _refresh_trt_status(self) -> None:
        """Re-check whether TRT engine is cached and update UI."""
        accel = coerce_enum(
            InferenceAccelerator,
            self.accelerator_combo.currentData(),
            InferenceAccelerator.AUTO,
        )
        if accel != InferenceAccelerator.TENSORRT:
            self.trt_deploy_btn.setVisible(False)
            if not self._trt_deploying:
                self.trt_status_label.setText("")
            return
        if self._trt_deploying:
            return
        self.trt_deploy_btn.setVisible(True)
        # Lightweight check: try import + check cache
        try:
            from clearvid.app.models.tensorrt_engine import (
                InferenceAccelerator as _IA,
                check_engine_ready as _check,
                find_compatible_engine as _find_compatible,
            )
            from clearvid.app.models.realesrgan_runner import (
                _MODEL_REGISTRY,
                _build_upsampler,
                ensure_realesrgan_weights,
                resolve_upscale_model,
            )
            from clearvid.app.bootstrap.paths import REALESRGAN_WEIGHTS_DIR, TRT_CACHE_DIR
            from clearvid.app.schemas.models import EnhancementConfig as _EC, QualityMode as _QM, UpscaleModel as _UM

            model_enum = coerce_enum(_UM, self.upscale_model_combo.currentData(), _UM.AUTO)
            # Pass accel=TENSORRT so that AUTO resolves to general_v3 (same as
            # the deploy button), instead of x4plus which has a different hash.
            model_key = resolve_upscale_model(model_enum, _QM.QUALITY, accel=_IA.TENSORRT)
            weights_dir = REALESRGAN_WEIGHTS_DIR
            if not (weights_dir / _MODEL_REGISTRY[model_key]["filename"]).exists():
                self.trt_status_label.setText("(请先下载模型权重)")
                self.trt_status_label.setStyleSheet(
                    "font-size: 11px; color: #888; padding-left: 4px;"
                )
                self.trt_deploy_btn.setEnabled(False)
                return

            tile = self.tile_size_spin.value() or 512
            batch = self._recommended_trt_batch()
            model_path = ensure_realesrgan_weights(weights_dir, model_key)
            dummy_config = _EC(
                input_path=Path("check"), output_path=Path("check"),
                tile_size=tile, batch_size=batch,
            )
            upsampler = _build_upsampler(dummy_config, model_path, model_key, tile, tile)
            ready, msg = _check(
                upsampler.model, fp16=True, tile_size=tile, batch_size=batch,
                cache_dir=TRT_CACHE_DIR, weight_path=model_path,
            )
            if ready:
                self.trt_status_label.setText(f"✓ 已就绪 (batch={batch})")
                self.trt_status_label.setStyleSheet(
                    "font-size: 11px; color: #4caf50; padding-left: 4px;"
                )
                self.trt_deploy_btn.setVisible(False)
            else:
                compat = _find_compatible(
                    upsampler.model, fp16=True, cache_dir=TRT_CACHE_DIR,
                    weight_path=model_path,
                )
                if compat is not None:
                    _, found_batch, _ = compat
                    self.trt_status_label.setText(
                        f"✓ 已就绪 (batch={found_batch}，建议部署 batch={batch})"
                    )
                    self.trt_status_label.setStyleSheet(
                        "font-size: 11px; color: #ff9800; padding-left: 4px;"
                    )
                    self.trt_deploy_btn.setEnabled(True)
                    self.trt_deploy_btn.setVisible(True)
                    self.trt_deploy_btn.setText(f"重新部署 batch={batch}")
                    return
                self.trt_deploy_btn.setText("部署 TensorRT 引擎")
                self.trt_status_label.setText(msg)
                self.trt_status_label.setStyleSheet(
                    "font-size: 11px; color: #ff9800; padding-left: 4px;"
                )
                self.trt_deploy_btn.setEnabled(True)
        except Exception:
            self.trt_status_label.setText("(无法检测状态)")
            self.trt_status_label.setStyleSheet(
                "font-size: 11px; color: #888; padding-left: 4px;"
            )

    def _recommended_trt_batch(self) -> int:
        raw_batch = self.batch_size_spin.value()
        if raw_batch > 0:
            return max(1, min(raw_batch, 4))
        try:
            import torch
            if torch.cuda.is_available():
                free_bytes, _ = torch.cuda.mem_get_info(0)
                free_gb = free_bytes / (1024 ** 3)
                if free_gb >= 8:
                    return 4
                if free_gb >= 4:
                    return 2
        except Exception:  # noqa: BLE001
            pass
        return 1

    def apply_recommendation(self, rec: Any) -> None:
        """Apply a Recommendation object to widget state."""
        set_combo_by_value(self.target_combo, rec.target_profile)
        set_combo_by_value(self.quality_combo, rec.quality_mode)
        set_combo_by_value(self.upscale_model_combo, rec.upscale_model)
        set_combo_by_value(self.accelerator_combo, rec.inference_accelerator)
        self.face_restore_enabled.setChecked(rec.face_restore_enabled)
        set_combo_by_value(self.face_model_combo, rec.face_restore_model)
        self.temporal_enabled.setChecked(rec.temporal_stabilize_enabled)
        self.sharpen_enabled.setChecked(rec.sharpen_enabled)
        self.sharpen_strength.setValue(rec.sharpen_strength)
        self.async_pipeline.setChecked(rec.async_pipeline)
        if hasattr(rec, "tile_size"):
            self.tile_size_spin.setValue(rec.tile_size)
        if hasattr(rec, "preprocess_denoise"):
            self.preprocess_denoise.setChecked(rec.preprocess_denoise)

    def set_progress(self, percent: int, message: str) -> None:
        self.progress_bar.setValue(max(0, min(100, percent)))
        self.progress_label.setText(message)

    def set_export_enabled(self, enabled: bool) -> None:
        self.export_btn.setEnabled(enabled)

    def set_exporting_state(self, active: bool) -> None:
        """Show/hide pause and cancel buttons based on export state.

        Also disables the TRT deploy button while an export is running to
        prevent VRAM exhaustion from two concurrent GPU-heavy tasks.
        """
        self._is_exporting = active
        self._pause_btn.setVisible(active)
        self._cancel_btn.setVisible(active)
        # Disable TRT deploy during export to prevent OOM
        if hasattr(self, "trt_deploy_btn"):
            if active:
                self.trt_deploy_btn.setEnabled(False)
                self.trt_deploy_btn.setToolTip("正在导出中，请等待导出完成后再部署 TensorRT 引擎")
            elif not self._trt_deploying:
                self.trt_deploy_btn.setEnabled(True)
                self.trt_deploy_btn.setToolTip("")
        if not active:
            self._is_paused = False
            self._pause_btn.setText("⏸ 暂停")

    def _toggle_pause(self) -> None:
        self._is_paused = not self._is_paused
        self._pause_btn.setText("▶ 继续" if self._is_paused else "⏸ 暂停")
        self.pause_requested.emit()

    def autofill_output(self, input_path: str, output_dir: str = "") -> None:
        """Auto-generate output path using the naming template."""
        if not input_path:
            return
        profile = coerce_enum(
            TargetProfile, self.target_combo.currentData(), TargetProfile.FHD
        )
        profile_val = profile.value if profile else "output"
        template = self.naming_edit.text() or DEFAULT_TEMPLATE
        filename = render_output_name(template, input_path, profile_val)
        out_dir = output_dir or str(OUTPUTS_DIR)
        self.output_edit.setText(str(Path(out_dir) / filename))

    def update_estimation(
        self,
        duration_sec: float,
        total_frames: int,
        source_size_bytes: int = 0,
    ) -> None:
        """Update the estimation label with rough time and size predictions."""
        # Store source metadata for re-computation when export duration changes.
        self._src_duration = duration_sec
        self._src_frames = total_frames
        self._src_size_bytes = source_size_bytes
        self._refresh_estimation()

    def _on_export_duration_changed(self) -> None:
        """Re-compute estimation when the user changes the export duration."""
        if hasattr(self, "_src_duration"):
            self._refresh_estimation()

    def _refresh_estimation(self) -> None:
        """(Re-)compute and display the estimation with current settings."""
        dur = self._src_duration
        frames = self._src_frames
        size_bytes = self._src_size_bytes

        # If user set a partial export duration, clamp to that.
        export_sec = self.preview_seconds.value()
        if export_sec > 0 and dur > 0:
            ratio = min(export_sec / dur, 1.0)
            dur = export_sec
            frames = max(1, int(frames * ratio))
            size_bytes = int(size_bytes * ratio)

        quality = coerce_enum(
            QualityMode, self.quality_combo.currentData(), QualityMode.QUALITY
        )
        profile = coerce_enum(
            TargetProfile, self.target_combo.currentData(), TargetProfile.FHD
        )
        crf_val = self.encoder_crf.value() if self.encoder_crf.value() > 0 else 18
        est = estimate_export(
            duration_sec=dur,
            total_frames=frames,
            quality_mode=quality.value if quality else "quality",
            target_profile=profile.value if profile else "fhd",
            encoder_crf=crf_val,
            source_size_bytes=size_bytes,
        )
        self.estimation_label.setText(f"📊 {est.description}")
        self.estimation_label.setVisible(True)
        self._last_estimate = est

    def show_post_export(self, output_path: str) -> None:
        """Show 'open folder' and 'play' buttons after successful export."""
        self._preview_progress_btn.setVisible(False)  # hide mid-export preview
        self._open_folder_btn.setVisible(True)
        self._play_btn.setVisible(True)
        # Disconnect previous connections (safe even if none exist)
        try:
            self._open_folder_btn.clicked.disconnect()
        except RuntimeError:
            pass
        try:
            self._play_btn.clicked.disconnect()
        except RuntimeError:
            pass
        self._open_folder_btn.clicked.connect(lambda: self._open_folder(output_path))
        self._play_btn.clicked.connect(lambda: self._play_video(output_path))

    def hide_post_export(self) -> None:
        self._open_folder_btn.setVisible(False)
        self._play_btn.setVisible(False)
        self._preview_progress_btn.setVisible(False)
        self._preview_progress_path = ""

    def update_preview_progress(self, preview_path: str) -> None:
        """Called when a mid-export preview file becomes available."""
        self._preview_progress_path = preview_path
        if not self._preview_progress_btn.isVisible():
            self._preview_progress_btn.setVisible(True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _browse_output(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "选择输出文件",
            self.output_edit.text() or str(OUTPUTS_DIR),
            "MP4 文件 (*.mp4);;所有文件 (*)",
        )
        if selected:
            self.output_edit.setText(selected)
            self.output_dir_changed.emit(str(Path(selected).parent))

    @staticmethod
    def _open_folder(path: str) -> None:
        """Open the containing folder and select the file (Windows)."""
        folder = str(Path(path).parent)
        if os.name == "nt":
            subprocess.Popen(  # noqa: S603, S607
                ["explorer", "/select,", os.path.normpath(path)]
            )
        else:
            subprocess.Popen(["xdg-open", folder])  # noqa: S603, S607

    @staticmethod
    def _play_video(path: str) -> None:
        """Open the video file with the system default player."""
        os.startfile(path)  # noqa: S606  # Windows-specific

    def _play_preview_progress(self) -> None:
        """Open the mid-export preview video with the system default player."""
        if self._preview_progress_path and os.path.isfile(self._preview_progress_path):
            os.startfile(self._preview_progress_path)  # noqa: S606  # Windows-specific
