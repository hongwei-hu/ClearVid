"""Persistent user settings using QSettings."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings


class UserSettings:
    """Manages persistent user preferences across sessions.

    Backed by Windows Registry (HKCU) via QSettings.
    """

    _MAX_RECENT = 10

    def __init__(self) -> None:
        self._s = QSettings("ClearVid", "ClearVid")

    # ---- Recent files ----

    def recent_files(self) -> list[str]:
        val = self._s.value("recent_files")
        if val is None:
            return []
        if isinstance(val, list):
            return [str(v) for v in val if v and Path(str(v)).exists()]
        if isinstance(val, str) and val:
            return [val] if Path(val).exists() else []
        return []

    def add_recent_file(self, path: str) -> None:
        recent = self.recent_files()
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)
        self._s.setValue("recent_files", recent[: self._MAX_RECENT])

    # ---- Directories ----

    def last_output_dir(self) -> str:
        return str(self._s.value("last_output_dir", "") or "")

    def set_last_output_dir(self, path: str) -> None:
        self._s.setValue("last_output_dir", path)

    def last_input_dir(self) -> str:
        return str(self._s.value("last_input_dir", "") or "")

    def set_last_input_dir(self, path: str) -> None:
        self._s.setValue("last_input_dir", path)

    # ---- Window geometry & state ----

    def window_geometry(self) -> object | None:
        return self._s.value("window_geometry")

    def save_window_geometry(self, data: object) -> None:
        self._s.setValue("window_geometry", data)

    def window_state(self) -> object | None:
        return self._s.value("window_state")

    def save_window_state(self, data: object) -> None:
        self._s.setValue("window_state", data)

    # ---- Splitter sizes ----

    def splitter_sizes(self) -> list[int] | None:
        val = self._s.value("splitter_sizes")
        if val and isinstance(val, list):
            try:
                return [int(v) for v in val]
            except (ValueError, TypeError):
                return None
        return None

    def save_splitter_sizes(self, sizes: list[int]) -> None:
        self._s.setValue("splitter_sizes", sizes)

    # ---- Panel collapse states ----

    def panel_states(self) -> dict[str, bool]:
        self._s.beginGroup("panel_states")
        result = {}
        for key in self._s.childKeys():
            val = self._s.value(key)
            if isinstance(val, bool):
                result[key] = val
            elif isinstance(val, str):
                result[key] = val.lower() == "true"
            else:
                result[key] = bool(val) if val is not None else False
        self._s.endGroup()
        return result

    def save_panel_state(self, name: str, expanded: bool) -> None:
        self._s.setValue(f"panel_states/{name}", expanded)

    # ---- Theme ----

    def theme(self) -> str:
        return str(self._s.value("theme", "dark") or "dark")

    def set_theme(self, name: str) -> None:
        self._s.setValue("theme", name)

    # ---- Naming template ----

    def naming_template(self) -> str:
        return str(self._s.value("naming_template", "{name}_{profile}") or "{name}_{profile}")

    def set_naming_template(self, template: str) -> None:
        self._s.setValue("naming_template", template)

    # ---- Notification preference ----

    def notify_on_complete(self) -> bool:
        val = self._s.value("notify_on_complete", True)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() == "true"
        return True

    def set_notify_on_complete(self, enabled: bool) -> None:
        self._s.setValue("notify_on_complete", enabled)

    # ---- Onboarding ----

    def onboarding_shown(self) -> bool:
        val = self._s.value("onboarding_shown", False)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() == "true"
        return False

    def set_onboarding_shown(self, shown: bool) -> None:
        self._s.setValue("onboarding_shown", shown)
