from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
import sys

def _resolve_assets_dir() -> Path:
    """Trouve le dossier assets le plus proche en remontant l'arborescence."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "assets"
        if candidate.exists():
            return candidate
    # Fallback : structure originale (src/bolt_app -> assets au-dessus)
    return here.parent.parent / "assets"


PACKAGE_DIR = Path(__file__).resolve().parent  # Bolt_app
ASSETS_DIR = _resolve_assets_dir()  # assets
PROJECT_ROOT = ASSETS_DIR.parent  # racine du projet ou livrable

DISPLAY_MATERIALS = (
    "Acier",
    "Aluminium",
    "Bronze",
    "Cuivre",
    "Inox",
    "Nylon",
    "PTFE",
    "Titane",
)  # Définition de l'ordre et des noms des matériaux affichés sur l'outil
ALLOWED_MATERIALS = {name.lower(): name for name in DISPLAY_MATERIALS}
MATERIAL_SYNONYMS = {
    "steel": "Acier",
    "gray cast iron": "Acier",
    "cast iron": "Acier",
    "bronze": "Bronze",
    "copper": "Cuivre",
    "brass": "Cuivre",
    "inox": "Inox",
    "stainless steel": "Inox",
    "aluminium": "Aluminium",
    "titane": "Titane",
    "nylon": "Nylon",
    "polyamide": "Nylon",
    "ptfe": "PTFE",
    "teflon": "PTFE",
}

PAS_STD_FILE = ASSETS_DIR / "Pas-std.csv"
FROTTEMENT_FILE = ASSETS_DIR / "Frottement.csv"
TROU_PASSAGE_FILE = ASSETS_DIR / "Trou_passage.csv"
TETE_VIS_FILE = ASSETS_DIR / "Tete_vis.csv"
FLOAT_TOLERANCE = 1e-6
DIMENSIONNEMENT_MARGIN = 0.08  # +8 % autorisé au-delà de Ft cible

__all__ = (
    "Vis",
    "DimensionnementResult",
    "dimensionner",
    "DISPLAY_MATERIALS",
    "ALLOWED_MATERIALS",
    "load_tete_vis_table",
    "lookup_diam_tete",
)

_TETE_TABLE_CACHE: Optional[Dict[float, Dict[str, float]]] = None
_HEAD_TYPES_CACHE: Tuple[str, ...] = ()

@dataclass
class Vis:
    diam_nominale: float
    diam_tete: float
    mat_vis: str
    serie: str = "H13"
    angle_filet: float = 60.0
    pas: float = field(init=False)

    mat_body: str = field(init=False, default="Acier")
    mat_sous_tete: str = field(init=False, default="Acier")
    is_lubrified: bool = field(init=False, default=False)
    is_sous_tete_lubrified: bool = field(init=False, default=False)
    diam_trou_passage: float = field(init=False)
    mu_filet: float = field(init=False, default=0.0)
    mu_sous_tete: float = field(init=False, default=0.0)
    last_denominator: float = field(init=False, default=0.0)
    pertes_frottements_filet: float = field(init=False, default=0.0)
    pertes_frottements_tete: float = field(init=False, default=0.0)
    pertes_frottements_totale: float = field(init=False, default=0.0)
    contrainte_traction: float = field(init=False, default=0.0)
    contrainte_torsion: float = field(init=False, default=0.0)
    contrainte_vm: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.mat_vis = self._normalize_material(self.mat_vis)
        self.serie = self._normalize_serie(self.serie)
        self.pas = self._load_pas_from_csv()
        self.diam_trou_passage = self._load_trou_passage_from_csv()
        self._check_dimensions()

    @property
    def diam_d1(self) -> float:
        return self.diam_nominale - 1.0825 * self.pas

    @property
    def diam_d2(self) -> float:
        return self.diam_nominale - 0.6495 * self.pas

    @property
    def diam_d3(self) -> float:
        return self.diam_nominale - (1.22687 * self.pas)

    @property
    def alpha(self) -> float:
        return self.angle_filet / 2.0

    def effort_serrage(
        self,
        couple: float,
        mat_body: Optional[str] = None,
        is_lubrified: bool = False,
        mat_sous_tete: Optional[str] = None,
        is_sous_tete_lubrified: bool = False,
    ) -> float:
        if couple < 0:
            raise ValueError("Le couple doit etre >= 0 N.mm.")

        self.mat_body = self._normalize_material(mat_body)
        if mat_sous_tete is None:
            self.mat_sous_tete = self.mat_body
        else:
            self.mat_sous_tete = self._normalize_material(mat_sous_tete)
        self.is_lubrified = bool(is_lubrified)
        self.is_sous_tete_lubrified = bool(is_sous_tete_lubrified)
        dh = (self.diam_tete + self.diam_trou_passage) / 2.0

        friction_filets = self._lookup_friction_coefficients(self.mat_vis, self.mat_body)
        if self.mat_sous_tete == self.mat_body:
            friction_sous_tete = friction_filets
        else:
            friction_sous_tete = self._lookup_friction_coefficients(self.mat_vis, self.mat_sous_tete)

        self.mu_filet = (
            friction_filets["lubricated_filet"]
            if self.is_lubrified
            else friction_filets["dry_filet"]
        )
        self.mu_sous_tete = (
            friction_sous_tete["lubricated_tete"]
            if self.is_sous_tete_lubrified
            else friction_sous_tete["dry_tete"]
        )

        denominateur = self._compute_denominator(dh)
        if denominateur == 0:
            raise ZeroDivisionError("Le denominateur du calcul est nul.")
        self.last_denominator = denominateur
        effort = couple / denominateur
        self._update_friction_losses(effort, couple, dh)
        self.contrainte_traction = self.calcul_contrainte_traction(effort)
        self.contrainte_torsion = self.calcul_contrainte_torsion(effort)
        self.contrainte_vm = self.contrainte_equivalent_VM(self.contrainte_traction, self.contrainte_torsion)
        return effort

    def _check_dimensions(self) -> None:
        if self.pas <= 0:
            raise ValueError("Le pas doit etre strictement positif (mm).")
        if min(self.diam_nominale, self.diam_tete) <= 0:
            raise ValueError("Les dimensions doivent etre strictement positives (mm).")
        if self.diam_tete < self.diam_nominale:
            raise ValueError("diametre de tete plus petit que le diametre nominale saisir une valeur valide")
        if self.diam_trou_passage <= 0:
            raise ValueError("Le diametre de trou de passage doit etre strictement positif (mm).")

    @staticmethod
    def _normalize_material(material: Optional[str], default_to_steel: bool = True) -> str:
        if material is None:
            if default_to_steel:
                return ALLOWED_MATERIALS["acier"]
            raise ValueError("Materiau non renseigne.")
        normalized = material.strip().lower()
        if normalized in ALLOWED_MATERIALS:
            return ALLOWED_MATERIALS[normalized]
        if normalized in MATERIAL_SYNONYMS:
            return MATERIAL_SYNONYMS[normalized]
        if default_to_steel:
            return ALLOWED_MATERIALS["acier"]
        raise ValueError(f"Materiau non supporte: {material}")

    @staticmethod
    def _normalize_serie(serie: Optional[str]) -> str:
        default_serie = "H13"
        if serie is None:
            return default_serie
        normalized = serie.strip().upper()
        if normalized in {"H12", "H13", "H14"}:
            return normalized
        return default_serie

    @staticmethod
    def _parse_float(value: str) -> float:
        text = value.strip().replace(",", ".")
        if not text or text == "-":
            raise ValueError("Valeur numerique manquante.")
        return float(text)

    def _compute_denominator(self, dh: float) -> float:
        return (0.161 * self.pas) + (self.mu_filet * self.diam_d2 / 1.715) + (self.mu_sous_tete * dh / 2.0)

    def _load_pas_from_csv(self) -> float:
        if not PAS_STD_FILE.exists():
            raise FileNotFoundError(f"Le fichier {PAS_STD_FILE} est introuvable.")
        with PAS_STD_FILE.open("r", encoding="utf-8", newline="") as csvfile:
            reader = csv.DictReader(csvfile, delimiter=";")
            for row in reader:
                raw_diam = (row.get("Diametre nominale") or "").strip()
                if not raw_diam:
                    continue
                try:
                    diam = self._parse_float(raw_diam)
                except ValueError:
                    continue
                if abs(diam - self.diam_nominale) <= FLOAT_TOLERANCE:
                    try:
                        return self._parse_float(row["Pas"])
                    except (KeyError, ValueError) as exc:
                        raise ValueError(
                            f"Pas introuvable pour le diametre nominal {self.diam_nominale} mm."
                        ) from exc
        raise ValueError(f"Diametre nominal {self.diam_nominale} mm non present dans Pas-std.csv.")

    def _load_trou_passage_from_csv(self) -> float:
        if not TROU_PASSAGE_FILE.exists():
            raise FileNotFoundError(f"Le fichier {TROU_PASSAGE_FILE} est introuvable.")
        with TROU_PASSAGE_FILE.open("r", encoding="utf-8", newline="") as csvfile:
            reader = csv.DictReader(csvfile, delimiter=";")
            for row in reader:
                raw_diam = (row.get("Diametre nominale") or "").strip()
                if not raw_diam:
                    continue
                try:
                    diam = self._parse_float(raw_diam)
                except ValueError:
                    continue
                if abs(diam - self.diam_nominale) <= FLOAT_TOLERANCE:
                    try:
                        raw_value = row[self.serie]
                    except KeyError as exc:
                        raise ValueError(
                            f"Serie {self.serie} absente pour le diametre {self.diam_nominale} mm."
                        ) from exc
                    try:
                        return self._parse_float(raw_value)
                    except ValueError as exc:
                        raise ValueError(
                            f"Valeur de diametre de trou de passage invalide pour {self.diam_nominale} mm ({self.serie})."
                        ) from exc
        raise ValueError(
            f"Diametre de trou de passage introuvable pour {self.diam_nominale} mm en serie {self.serie}."
        )

    def _lookup_friction_coefficients(self, mat_vis: str, mat_body: str) -> dict[str, float]:
        if not FROTTEMENT_FILE.exists():
            raise FileNotFoundError(f"Le fichier {FROTTEMENT_FILE} est introuvable.")
        with FROTTEMENT_FILE.open("r", encoding="utf-8", newline="") as csvfile:
            reader = csv.DictReader(csvfile, delimiter=";")
            if reader.fieldnames:
                reader.fieldnames = [name.strip() if isinstance(name, str) else name for name in reader.fieldnames]
            for row in reader:
                row = {key.strip(): value for key, value in row.items() if key}
                try:
                    row_mat_vis = self._normalize_material(row.get("Material 1"), default_to_steel=False)
                    row_mat_body = self._normalize_material(row.get("Material 2"), default_to_steel=False)
                except ValueError:
                    continue
                if row_mat_vis == mat_vis and row_mat_body == mat_body:
                    try:
                        return {
                            "dry_filet": self._parse_float(row["Dry_max_filet"]),
                            "dry_tete": self._parse_float(row["Dry_max_tete"]),
                            "lubricated_filet": self._parse_float(row["Lubricated_max_filet"]),
                            "lubricated_tete": self._parse_float(row["Lubricated_max_tete"]),
                        }
                    except (KeyError, ValueError) as exc:
                        raise ValueError(
                            f"Coefficient de friction incomplet pour {mat_vis}/{mat_body}."
                        ) from exc
        raise ValueError(f"Aucun coefficient de friction pour {mat_vis}/{mat_body} dans Frottement.csv.")

    def _update_friction_losses(self, effort: float, couple: float, dh: float) -> None:
        if couple <= 0:
            self.pertes_frottements_filet = 0.0
            self.pertes_frottements_tete = 0.0
            self.pertes_frottements_totale = 0.0
            return
        filets_torque = effort * (self.mu_filet * (self.diam_d2 / 1.715))
        tete_torque = effort * (self.mu_sous_tete * (dh / 2.0))
        pertes_filet = (filets_torque / couple) * 100.0
        pertes_tete = (tete_torque / couple) * 100.0
        self.pertes_frottements_filet = pertes_filet
        self.pertes_frottements_tete = pertes_tete
        self.pertes_frottements_totale = pertes_filet + pertes_tete

    def calcul_contrainte_traction(self, effort_serrage: float, diam_nominale: Optional[float] = None) -> float:
        diam = self.diam_nominale if diam_nominale is None else float(diam_nominale)
        aire_traction = (math.pi / 4.0) * ((diam - (0.9392 * self.pas)) ** 2)
        if aire_traction <= 0:
            raise ValueError("La surface de traction est invalide.")
        return effort_serrage / aire_traction

    def calcul_contrainte_torsion(self, effort_serrage: float) -> float:
        if self.diam_d2 <= 0 or self.diam_d3 <= 0:
            raise ValueError("Les diametres internes doivent etre strictement positifs.")
        tan_phi = self.pas / (math.pi * self.diam_d2)
        cos_alpha = math.cos(math.radians(self.alpha))
        if cos_alpha == 0:
            raise ZeroDivisionError("Le cosinus de l'angle alpha est nul.")
        tan_rho = self.mu_filet / cos_alpha
        denominateur = 1.0 - (tan_phi * tan_rho)
        if denominateur == 0:
            raise ZeroDivisionError("Le denominateur du calcul de torsion est nul.")
        mth = effort_serrage * (self.diam_d2 / 2.0) * ((tan_phi + tan_rho) / denominateur)
        return (16.0 * mth) / (math.pi * (self.diam_d3 ** 3))

    def contrainte_equivalent_VM(self, contrainte_traction: float, contrainte_torsion: float) -> float:
        return math.sqrt((contrainte_traction ** 2) + (3.0 * (contrainte_torsion ** 2)))

def _candidate_tete_paths(initial: Optional[Path] = None) -> List[Path]:
    candidates: List[Path] = []
    if initial:
        candidates.append(initial)
    candidates.append(TETE_VIS_FILE)
    try:
        exe_dir = Path(sys.argv[0]).resolve().parent
        candidates.append(exe_dir / "assets" / "Tete_vis.csv")
    except Exception:
        pass
    try:
        meipass = Path(getattr(sys, "_MEIPASS"))  # type: ignore[attr-defined]
        candidates.append(meipass / "assets" / "Tete_vis.csv")
    except Exception:
        pass
    try:
        candidates.append(Path(__file__).resolve().parents[2] / "assets" / "Tete_vis.csv")
    except Exception:
        pass
    seen = set()
    unique: List[Path] = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def load_tete_vis_table(path: Optional[Path] = None) -> Tuple[Dict[float, Dict[str, float]], Tuple[str, ...]]:
    """Charge Tete_vis.csv et retourne un mapping diametre -> {type de tete: diametre} et la liste des types."""
    file_path: Optional[Path] = None
    for candidate in _candidate_tete_paths(path):
        if candidate and candidate.exists():
            file_path = candidate
            break
    if file_path is None:
        raise FileNotFoundError(f"Le fichier {path or TETE_VIS_FILE} est introuvable.")

    table: Dict[float, Dict[str, float]] = {}
    head_types: List[str] = []
    with file_path.open("r", encoding="utf-8", newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        header: List[str] = []
        for idx, row in enumerate(reader):
            if idx == 0:
                header = [cell.strip() for cell in row]
                head_types = [cell for cell in header[1:] if cell]
                continue
            if not row or not row[0].strip():
                continue
            try:
                diam = float(row[0].replace(",", "."))
            except ValueError:
                continue
            values: Dict[str, float] = {}
            for col_idx, head_name in enumerate(head_types, start=1):
                cell = row[col_idx] if col_idx < len(row) else ""
                cell = (cell or "").strip()
                if not cell or cell == "-":
                    continue
                try:
                    values[head_name] = float(cell.replace(",", "."))
                except ValueError:
                    continue
            table[diam] = values
    return table, tuple(head_types)


def _get_tete_table() -> Tuple[Dict[float, Dict[str, float]], Tuple[str, ...]]:
    global _TETE_TABLE_CACHE, _HEAD_TYPES_CACHE
    if _TETE_TABLE_CACHE is not None:
        return _TETE_TABLE_CACHE, _HEAD_TYPES_CACHE
    table, head_types = load_tete_vis_table()
    _TETE_TABLE_CACHE = table
    _HEAD_TYPES_CACHE = head_types
    return table, head_types

def lookup_diam_tete(diam_nominale: float, head_type: str) -> Optional[float]:
    """Retourne le diametre de tete ISO pour un type et un diametre nominal donnes, ou None si non ISO."""
    table, _ = _get_tete_table()
    target_head = head_type.strip()
    for diam, values in table.items():
        if abs(diam - diam_nominale) <= FLOAT_TOLERANCE:
            return values.get(target_head)
    return None

@dataclass
class DimensionnementResult:
    diam_nominale: float
    diam_tete: float
    serie: str
    effort: float
    couple: float
    mat_vis: str
    mat_body: str
    mat_sous_tete: str
    lubrified: bool
    head_types: List[str] = field(default_factory=list)

def dimensionner(
    diametres: List[float],
    mat_vis: str,
    mat_body: str,
    effort_cible: float,
    couple_max: float,
    *,
    serie: str = "H13",
    mat_sous_tete: Optional[str] = None,
    include_lubrified: bool = False,
    tete_table: Optional[Dict[float, Dict[str, float]]] = None,
    manual_diam_tete: Optional[float] = None,
) -> List[DimensionnementResult]:
    if effort_cible <= 0:
        raise ValueError("L'effort de serrage cible doit etre strictement positif.")
    if couple_max <= 0:
        raise ValueError("Le couple maximal doit etre strictement positif.")
    if not diametres:
        raise ValueError("Aucun diametre nominal fourni.")

    table, _ = _get_tete_table()
    if tete_table is None:
        tete_table = table

    diametres_uniques = sorted(set(diametres))

    mat_sous_tete_effectif = mat_sous_tete
    if not mat_sous_tete_effectif:
        mat_sous_tete_effectif = mat_body

    iso_available = any(tete_table.get(diam) for diam in diametres_uniques)
    use_manual = not iso_available
    manual_value = manual_diam_tete if use_manual else None
    if use_manual:
        if manual_value is None:
            raise ValueError("Aucune valeur ISO disponible pour cette plage, veuillez saisir un diametre de tete.")
        if manual_value <= 0 or manual_value >= 120:
            raise ValueError("Le diametre de tete saisi doit etre positif et strictement inferieur a 120 mm.")
        if manual_value <= max(diametres_uniques):
            raise ValueError("diametre de tete plus petit que le diametre nominale saisir une valeur valide")

    solutions: Dict[tuple[float, bool], DimensionnementResult] = {}
    head_sets: Dict[tuple[float, bool], set[str]] = {}

    def _evaluate(diam: float, lub: bool) -> None:
        heads = tete_table.get(diam, {})
        candidates = list(heads.items()) if not use_manual else [("Saisie manuelle", manual_value)]  # type: ignore[list-item]
        if not candidates:
            return
        for head_type, diam_tete in candidates:
            if diam_tete is None:
                continue
            if diam_tete <= diam:
                continue
            vis = Vis(diam_nominale=diam, diam_tete=diam_tete, mat_vis=mat_vis, serie=serie)
            _effort_max = vis.effort_serrage(
                couple_max,
                mat_body=mat_body,
                is_lubrified=lub,
                mat_sous_tete=mat_sous_tete_effectif,
                is_sous_tete_lubrified=False,
            )
            dh = (vis.diam_tete + vis.diam_trou_passage) / 2.0
            denom = vis.last_denominator or vis._compute_denominator(dh)  # type: ignore[attr-defined]

            couple_min_requis = effort_cible * denom
            couple_max_permis = min(couple_max, effort_cible * (1.0 + DIMENSIONNEMENT_MARGIN) * denom)
            if couple_max_permis + FLOAT_TOLERANCE < couple_min_requis:
                continue

            couple_recommande = couple_min_requis * (1.0 + DIMENSIONNEMENT_MARGIN / 2.0)
            if couple_recommande > couple_max_permis:
                couple_recommande = couple_max_permis
            effort_recommande = couple_recommande / denom

            if effort_recommande + FLOAT_TOLERANCE < effort_cible:
                continue
            if effort_recommande - FLOAT_TOLERANCE > effort_cible * (1.0 + DIMENSIONNEMENT_MARGIN):
                continue

            key = (diam, lub)
            head_sets.setdefault(key, set()).add(head_type)
            previous = solutions.get(key)
            if previous is None or couple_recommande < previous.couple - FLOAT_TOLERANCE:
                solutions[key] = DimensionnementResult(
                    diam_nominale=diam,
                    diam_tete=vis.diam_tete,
                    serie=vis.serie,
                    effort=effort_recommande,
                    couple=couple_recommande,
                    mat_vis=vis.mat_vis,
                    mat_body=vis.mat_body,
                    mat_sous_tete=vis.mat_sous_tete,
                    lubrified=lub,
                )

    for diam in diametres_uniques:
        _evaluate(diam, False)
        if include_lubrified:
            _evaluate(diam, True)

    results: List[DimensionnementResult] = []
    for key, res in solutions.items():
        head_types = sorted(head_sets.get(key, []))
        res.head_types = head_types
        results.append(res)

    return sorted(results, key=lambda res: (res.diam_nominale, res.lubrified, res.couple))

# Ici le main n'est pas utile.
def main() -> None:
    vis_1 = Vis(10.0, 13.0, "Acier")
    effort = vis_1.effort_serrage(24000.0, mat_body="Acier")
    print(f"Effort de serrage: {effort:.2f} N")
    print(f"diam_d1: {vis_1.diam_d1:.2f} mm")
    print(f"diam_d2: {vis_1.diam_d2:.2f} mm")

if __name__ == "__main__":
    main()
