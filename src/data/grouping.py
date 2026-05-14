"""Group stems by CV unit for leave-one-group-out evaluation."""


def stem_to_group(stem, metadata, cv_unit, task):
    """Return a group id, or None if the stem has no group for this cv_unit.

    cv_unit:
        "volume"    -> stem itself (one group per vol; equivalent to current LOO)
        "replicate" -> "{label}_tissue={Tissue}". Tissue ints are reused across
                       conditions, so the task label (Exercise / Perturbation)
                       is concatenated to make tissue ids unique. Per colleague:
                       "Concatenate Perturbation + Tissue columns".
    """
    if cv_unit == "volume":
        return stem
    if cv_unit == "replicate":
        t = metadata.get(stem, {}).get("Tissue")
        if t in (None, ""):
            return None
        label_col = "Exercise" if task == "exercise" else "Perturbation"
        label = metadata.get(stem, {}).get(label_col)
        if label in (None, ""):
            return None
        return f"{label}_tissue={t}"
    raise ValueError(f"unknown cv_unit={cv_unit}")
