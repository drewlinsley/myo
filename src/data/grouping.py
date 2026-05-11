"""Group stems by CV unit for leave-one-group-out evaluation."""

import re

DATE_RE = re.compile(r"(?<![0-9])(\d{6})(?![0-9])")


def stem_to_group(stem, metadata, cv_unit, task):
    """Return a group id, or None if the stem has no group for this cv_unit.

    cv_unit:
        "volume"    -> stem itself (one group per vol; equivalent to current LOO)
        "replicate" -> "{task}_tissue={Tissue}"  (Tissue is scoped per-task)
        "date"      -> 6-digit MMDDYY parsed from the stem
    """
    if cv_unit == "volume":
        return stem
    if cv_unit == "replicate":
        t = metadata.get(stem, {}).get("Tissue")
        if t in (None, ""):
            return None
        return f"{task}_tissue={t}"
    if cv_unit == "date":
        m = DATE_RE.search(stem)
        return m.group(1) if m else None
    raise ValueError(f"unknown cv_unit={cv_unit}")
