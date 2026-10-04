"""Normalising a search space at the moment it is created.

The instruction this implements: "Whenever the search space is initiated, first
thing which you have to do is we have to normalize the search space to our
knowledge bank and give it to the user saying, 'this is what the normalization
is, and I'm going with this knowledge.'"

So every normalisation is REPORTED, never silent. The caller gets back exactly
what was changed and what could not be resolved.

Two rules this obeys, both from standing project policy:

  * Resolution is DETERMINISTIC. RDKit canonicalisation only. No LLM call, no
    network lookup (the NIH/CACTUS resolver EDBO uses is both non-deterministic
    and unavailable offline).
  * Ambiguity is FLAGGED, never guessed. A name we cannot resolve from a local
    source is passed through verbatim and listed for human review.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

SMILES_HINT = set("()[]=#@+-\\/123456789%")


@dataclass
class NormalisationReport:
    """What was done, and what needs a human."""
    canonicalised: List[Dict[str, str]] = field(default_factory=list)
    unchanged: int = 0
    unresolved: List[Dict[str, str]] = field(default_factory=list)
    collisions: List[Dict[str, Any]] = field(default_factory=list)
    backend: str = "none"

    @property
    def needs_review(self) -> bool:
        return bool(self.unresolved or self.collisions)

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "n_canonicalised": len(self.canonicalised),
            "n_unchanged": self.unchanged,
            "canonicalised": self.canonicalised,
            "unresolved": self.unresolved,
            "collisions": self.collisions,
            "needs_review": self.needs_review,
            "summary": self.summary(),
        }

    def summary(self) -> str:
        bits = [f"{len(self.canonicalised)} level(s) canonicalised",
                f"{self.unchanged} already canonical"]
        if self.unresolved:
            bits.append(f"{len(self.unresolved)} NOT resolved (passed through "
                        f"verbatim, listed for review)")
        if self.collisions:
            bits.append(f"{len(self.collisions)} COLLISION(S): distinct inputs "
                        f"that denote the same structure")
        return "; ".join(bits)


def _rdkit():
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        return Chem
    except Exception:          # noqa: BLE001
        return None


def looks_like_smiles(s: str) -> bool:
    """Only a hint for reporting. The authority is whether RDKit parses it."""
    return isinstance(s, str) and bool(s) and bool(SMILES_HINT & set(s))


def canonical(smiles: str) -> Optional[str]:
    Chem = _rdkit()
    if Chem is None:
        return None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return Chem.MolToSmiles(m)


def normalise_space(space_dict: dict) -> Tuple[dict, NormalisationReport]:
    """Canonicalise every chemical level in a search-space definition.

    Returns (normalised copy, report). The input is not mutated.
    """
    import copy

    out = copy.deepcopy(space_dict)
    rep = NormalisationReport()
    Chem = _rdkit()
    rep.backend = "rdkit" if Chem is not None else "none (rdkit unavailable)"

    if Chem is None:
        # Better to say so than to pretend the space was normalised.
        for f in out.get("factors", []):
            for lv in (f.get("levels") or []):
                if isinstance(lv, str) and looks_like_smiles(lv):
                    rep.unresolved.append({
                        "factor": f.get("name"), "value": lv,
                        "reason": "rdkit not available in this environment; "
                                  "level passed through unchanged"})
        return out, rep

    for f in out.get("factors", []):
        levels = f.get("levels")
        if not levels:
            continue
        seen: Dict[str, str] = {}
        new_levels, remap = [], {}

        for lv in levels:
            if not isinstance(lv, str):
                new_levels.append(lv)
                rep.unchanged += 1
                continue

            can = canonical(lv)
            if can is None:
                # Do not guess which it is. Report both possibilities and the
                # fact that nothing was changed.
                rep.unresolved.append({
                    "factor": f.get("name"), "value": lv,
                    "reason": ("could not be parsed as SMILES. It is either a "
                               "chemical name or malformed SMILES; no "
                               "deterministic offline name resolver is "
                               "configured, so it was passed through verbatim "
                               "and is NOT matched against the other levels"),
                    "looks_like_smiles": looks_like_smiles(lv)})
                new_levels.append(lv)
                continue

            if can != lv:
                rep.canonicalised.append({"factor": f.get("name"),
                                          "from": lv, "to": can})
                remap[lv] = can
            else:
                rep.unchanged += 1

            if can in seen and seen[can] != lv:
                # Two inputs the user believes are different compounds are the
                # same structure. Silently deduplicating would shrink the space
                # under them; raising is for the caller to decide.
                rep.collisions.append({
                    "factor": f.get("name"), "canonical": can,
                    "inputs": [seen[can], lv]})
            seen.setdefault(can, lv)
            new_levels.append(can)

        f["levels"] = new_levels

        # descriptors are keyed by level, so they move with it
        if remap and isinstance(f.get("descriptors"), dict):
            f["descriptors"] = {remap.get(k, k): v
                                for k, v in f["descriptors"].items()}

    return out, rep
