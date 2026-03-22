"""Output file naming template system with variable substitution."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


# Default template
DEFAULT_TEMPLATE = "{name}_{profile}"


def render_output_name(
    template: str,
    input_path: str | Path,
    profile: str = "output",
    *,
    ext: str = ".mp4",
) -> str:
    """Render a naming template to a concrete filename.

    Supported variables:
        {name}    — input filename stem (without extension)
        {profile} — target profile value (e.g. fhd, uhd4k)
        {date}    — current date YYYY-MM-DD
        {time}    — current time HHmmss
        {ext}     — original input extension (without dot)

    Returns the rendered filename **without** directory prefix.
    """
    inp = Path(input_path)
    now = datetime.now()  # noqa: DTZ005
    variables = {
        "name": inp.stem,
        "profile": profile,
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H%M%S"),
        "ext": inp.suffix.lstrip(".") or "mp4",
    }

    # Do a safe format: unknown keys stay as-is
    result = template
    for key, val in variables.items():
        result = result.replace("{" + key + "}", val)

    # Ensure we have an output extension
    if not result.endswith(ext):
        result += ext

    return result
