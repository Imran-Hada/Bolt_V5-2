from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import streamlit as st

# Local paths (self contained in the livrables folder)
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
ASSETS_DIR = BASE_DIR / "assets"

if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from bolt_app.core import DISPLAY_MATERIALS, Vis, dimensionner, load_tete_vis_table

APP_VERSION = "v5.2"
LOGO_PATH = ASSETS_DIR / "Bolt_logo.png"
TECH_IMAGE = ASSETS_DIR / "Schema_technique.png"
CONSTRUCTION_IMAGE = ASSETS_DIR / "construction_image.png"
LOGO_WIDTH = 220  # ~40% wider than previous display
IMAGE_COLUMNS = (5, 6)  # Wider image column for both tabs (~45% increase)

DEFAULT_HEAD_TYPES = [
    "Hexagonale",
    "CHC cylindrique",
    "CHC fraisee",
    "CHC bombee",
    "Bombee fendue",
    "Bombee Philips",
    "Bombee Pozidriv",
    "Bombee Torx interne",
]
HEAD_LABEL_MAP: Dict[str, str] = {
    "CHC tete fraisee": "CHC fraisee",
    "CHC tete bombee": "CHC bombee",
    "Tete bombee fendue": "Bombee fendue",
    "Tete bombee philips": "Bombee Philips",
    "Tete bombee Pozidriv": "Bombee Pozidriv",
    "Tete bombee Torx interne": "Bombee Torx interne",
}
HEAD_REVERSE_MAP: Dict[str, str] = {v: k for k, v in HEAD_LABEL_MAP.items()}


@st.cache_data
def load_diametres(path: Path) -> List[float]:
    values: List[float] = []
    with path.open("r", encoding="utf-8", newline="") as fichier:
        reader = csv.DictReader(fichier, delimiter=";")
        for row in reader:
            cell = (row.get("Diametre nominale") or "").strip()
            if not cell or cell == "-":
                continue
            try:
                values.append(float(cell.replace(",", ".")))
            except ValueError:
                continue
    return sorted(set(values))


@st.cache_data
def load_tete_table() -> Tuple[Dict[float, Dict[str, float]], Tuple[str, ...], str | None]:
    try:
        table, head_types = load_tete_vis_table()
        return table, head_types, None
    except Exception as exc:  # pragma: no cover - defensive for missing file on Streamlit
        return {}, tuple(DEFAULT_HEAD_TYPES), str(exc)


def display_head_name(raw_name: str) -> str:
    return HEAD_LABEL_MAP.get(raw_name, raw_name)


def raw_head_name(display_name: str) -> str:
    return HEAD_REVERSE_MAP.get(display_name, display_name)


def format_couple_for_display(value: float, unit: str) -> str:
    return f"{value / 1000.0:.2f}" if unit == "N.m" else f"{value:.0f}"


def resolve_logo() -> str | None:
    if LOGO_PATH.exists():
        return str(LOGO_PATH)
    return None


st.set_page_config(
    page_title=f"Bolt {APP_VERSION} - Effort de serrage",
    page_icon=resolve_logo(),
    layout="wide",
)

logo_col, title_col = st.columns([2, 5])
with logo_col:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=LOGO_WIDTH)
with title_col:
    st.title(f"Bolt {APP_VERSION} - Effort de serrage")
    st.write("Calcul et dimensionnement de visserie en mode Streamlit.")

tete_table, head_types_raw, tete_error = load_tete_table()
if tete_error:
    st.warning(f"Table des tetes ISO indisponible: {tete_error}")

diam_values = load_diametres(ASSETS_DIR / "Pas-std.csv")
materials = sorted(DISPLAY_MATERIALS)
head_type_options = [display_head_name(name) for name in head_types_raw] if head_types_raw else list(DEFAULT_HEAD_TYPES)

tab_calc, tab_dim = st.tabs(["Calcul direct", "Dimensionnement"])


def render_image(image_path: Path) -> None:
    if image_path.exists():
        st.image(str(image_path), use_column_width=True)
    else:
        st.info("Image indisponible dans les assets.")


with tab_calc:
    st.subheader("Couple -> Effort de serrage et contraintes")
    col_form, col_img = st.columns(IMAGE_COLUMNS, gap="large")
    with col_form:
        with st.form("calc_form"):
            dn = st.selectbox(
                "Diametre nominal (M-)",
                options=diam_values or [12.0],
                index=(diam_values or [12.0]).index(12.0) if 12.0 in diam_values else 0,
            )
            head_display = st.selectbox("Type de tete", options=head_type_options)
            raw_head = raw_head_name(head_display)
            dh_iso = tete_table.get(dn, {}).get(raw_head)
            manual_dh: float | None = None
            if dh_iso is None:
                st.info("Pas de valeur ISO pour cette combinaison. Saisissez un diametre de tete.")
                manual_dh = st.number_input("Diametre tete de vis (mm)", min_value=0.0, step=0.1)
            else:
                st.caption(f"Diametre tete de vis ISO: {dh_iso:g} mm")

            st.divider()
            st.write("Materiaux et conditions")
            mat_vis = st.selectbox("Materiau de la vis", options=materials, index=materials.index("Acier") if "Acier" in materials else 0)
            mat_body = st.selectbox("Materiau de la piece", options=materials, index=materials.index("Acier") if "Acier" in materials else 0)
            is_filet_lub = st.checkbox("Lubrification des filets", value=False)
            has_under_head = st.checkbox("Materiau different sous tete", value=False)
            mat_sous_tete = mat_body
            if has_under_head:
                mat_sous_tete = st.selectbox("Materiau sous tete", options=materials, index=materials.index(mat_body))
            is_under_head_lub = st.checkbox("Lubrification sous tete", value=False, disabled=not has_under_head)

            st.divider()
            st.write("Chargement")
            couple = st.number_input("Couple applique", min_value=0.0, value=40.0, step=1.0)
            unit = st.selectbox("Unite du couple", options=["N.m", "N.mm"], index=0)

            submitted = st.form_submit_button("Calculer")

        if submitted:
            try:
                dh_value = dh_iso if dh_iso is not None else manual_dh
                if dh_value is None:
                    st.error("Veuillez saisir un diametre de tete de vis.")
                    st.stop()
                if dh_value <= 0 or dh_value >= 120:
                    st.error("Le diametre de tete doit etre positif et inferieur a 120 mm.")
                    st.stop()
                if dh_value <= dn:
                    st.error("Le diametre de tete doit etre superieur au diametre nominal.")
                    st.stop()

                couple_mm = couple * 1000.0 if unit == "N.m" else couple
                vis = Vis(diam_nominale=dn, diam_tete=dh_value, mat_vis=mat_vis)
                effort = vis.effort_serrage(
                    couple_mm,
                    mat_body=mat_body,
                    is_lubrified=is_filet_lub,
                    mat_sous_tete=mat_sous_tete if has_under_head else None,
                    is_sous_tete_lubrified=is_under_head_lub if has_under_head else False,
                )

                st.success("Calcul termine.")
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                metrics_col1.metric("Effort de serrage Ft (N)", f"{effort:.0f}")
                metrics_col2.metric("Contrainte de traction (MPa)", f"{vis.contrainte_traction:.1f}")
                metrics_col3.metric("Contrainte de torsion (MPa)", f"{vis.contrainte_torsion:.1f}")
                vm_col, loss_filet_col, loss_tete_col = st.columns(3)
                vm_col.metric("Contrainte equivalente VM (MPa)", f"{vis.contrainte_vm:.1f}")
                loss_filet_col.metric("Pertes filets (%)", f"{vis.pertes_frottements_filet:.0f}")
                loss_tete_col.metric("Pertes sous tete (%)", f"{vis.pertes_frottements_tete:.0f}")
                st.caption(f"Pertes totales: {vis.pertes_frottements_totale:.0f} %")
            except Exception as exc:  # pragma: no cover - Streamlit feedback
                st.error(str(exc))

    with col_img:
        render_image(TECH_IMAGE)


with tab_dim:
    st.subheader("Dimensionnement (effort cible -> choix de vis)")
    col_form, col_img = st.columns(IMAGE_COLUMNS, gap="large")
    with col_form:
        with st.form("dim_form"):
            if diam_values:
                diam_min, diam_max = st.select_slider(
                    "Plage de diametres (M-)",
                    options=diam_values,
                    value=(diam_values[0], diam_values[min(len(diam_values) - 1, 5)]),
                )
            else:
                diam_min, diam_max = 8.0, 12.0
                st.warning("Impossible de lire Pas-std.csv, valeurs par defaut utilisees.")

            iso_available = any(tete_table.get(d) for d in diam_values if diam_min - 1e-6 <= d <= diam_max + 1e-6)
            manual_dh_dim: float | None = None
            if not iso_available:
                st.info("Aucune tete ISO dans cette plage, saisissez un diametre de tete.")
                manual_dh_dim = st.number_input("Diametre tete de vis (mm)", min_value=0.0, step=0.1)

            st.divider()
            st.write("Materiaux et conditions")
            mat_vis_dim = st.selectbox("Materiau de la vis", options=materials, key="mat_vis_dim")
            mat_body_dim = st.selectbox("Materiau de la piece", options=materials, key="mat_body_dim")
            has_under_head_dim = st.checkbox("Materiau different sous tete", value=False, key="under_head_dim")
            mat_sous_tete_dim = mat_body_dim
            if has_under_head_dim:
                mat_sous_tete_dim = st.selectbox("Materiau sous tete", options=materials, key="mat_sous_tete_dim")
            is_filet_lub_dim = st.checkbox("Lubrification des filets", value=False, key="filet_lub_dim")

            st.divider()
            st.write("Chargement")
            effort_cible = st.number_input("Effort de serrage cible (N)", min_value=0.0, value=10000.0, step=500.0)
            couple_max = st.number_input("Couple maximal constructeur", min_value=0.0, value=40.0, step=1.0)
            unit_dim = st.selectbox("Unite du couple", options=["N.m", "N.mm"], index=0, key="unit_dim")

            submitted_dim = st.form_submit_button("Trouver des configurations")

        if submitted_dim:
            try:
                diam_range = [d for d in diam_values if diam_min - 1e-6 <= d <= diam_max + 1e-6] or [diam_min, diam_max]
                manual_value = manual_dh_dim if not iso_available else None
                couple_max_mm = couple_max * 1000.0 if unit_dim == "N.m" else couple_max

                results = dimensionner(
                    diametres=diam_range,
                    mat_vis=mat_vis_dim,
                    mat_body=mat_body_dim,
                    mat_sous_tete=mat_sous_tete_dim if has_under_head_dim else None,
                    effort_cible=effort_cible,
                    couple_max=couple_max_mm,
                    include_lubrified=is_filet_lub_dim,
                    tete_table=tete_table,
                    manual_diam_tete=manual_value,
                )

                if not results:
                    if is_filet_lub_dim:
                        st.warning("Pas de solution pour cette configuration.")
                    else:
                        st.warning("Pas de resultat. Essayez avec lubrification.")
                    st.stop()

                unit_label = unit_dim
                display_rows = []
                for res in results:
                    couple_display = format_couple_for_display(res.couple, unit_label)
                    head_display = ", ".join(display_head_name(name) for name in res.head_types) if res.head_types else "-"
                    display_rows.append(
                        {
                            "Diametre (M-)": f"M{res.diam_nominale:g}",
                            "Couple": f"{couple_display} {unit_label}",
                            "Effort (N)": f"{res.effort:.0f}",
                            "Etat": "Lubrifie" if res.lubrified else "Sec",
                            "Sous tete": res.mat_sous_tete,
                            "Serie": res.serie,
                            "Types de tete": head_display,
                        }
                    )

                st.success(f"{len(display_rows)} configuration(s) trouvee(s).")
                st.dataframe(display_rows, use_container_width=True)
            except Exception as exc:  # pragma: no cover - Streamlit feedback
                st.error(str(exc))

    with col_img:
        render_image(CONSTRUCTION_IMAGE)
