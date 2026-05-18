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
        if task == "exercise":
            label = metadata.get(stem, {}).get("Exercise")
        else:
            # Prefer the original (uncollapsed) Perturbation label when present
            # so dose variants don't get bundled into a single physical tissue.
            label = (metadata.get(stem, {}).get("_perturbation_orig")
                     or metadata.get(stem, {}).get("Perturbation"))
        if label in (None, ""):
            return None
        return f"{label}_tissue={t}"
    raise ValueError(f"unknown cv_unit={cv_unit}")
