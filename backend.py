import pandas as pd
import streamlit as st
from scipy import stats as sci_stats
import numpy as np
from difflib import get_close_matches
import unicodedata


def extraer_columnas_validas(df_encuesta):
    """
    Busca las columnas más parecidas a las esperadas en el DataFrame recibido.
    Devuelve un diccionario con alias y nombres reales.
    """
    columnas_esperadas = {
        "gasto_evento": "¿Cuánto ha gastado aproximadamente en actividades relacionadas con LOS EVENTOS RELIGIOSOS DE SEMANA SANTA EN CARTAGENA (souvenirs, artesanías, libros, etc.)?",
        "dias_estadia": "¿Cuántos días estará en la ciudad donde se desarrolla este evento?",
        "gasto_alojamiento": "¿Cuánto está gastando gasto diariamente en alojamiento? (Por persona):",
        "gasto_alimentacion": "En promedio ¿Cuánto ha sido su gasto diario en alimentación y bebidas durante su estadía en la ciudad?",
        "gasto_transporte": "En promedio ¿Cuánto ha sido su gasto diario en transporte durante su estadía en la ciudad?"
    }

    mapeo_resultante = {}
    columnas_disponibles = df_encuesta.columns.tolist()

    for alias, col_esperada in columnas_esperadas.items():
        coincidencias = get_close_matches(col_esperada.strip(), columnas_disponibles, n=1, cutoff=0.7)
        if coincidencias:
            mapeo_resultante[alias] = coincidencias[0]
        else:
            mapeo_resultante[alias] = None  # No encontrada

    return mapeo_resultante


def detectar_categorias_motivo(
    df_encuesta: pd.DataFrame,
    columna_reside: str = "¿Reside en la ciudad donde se desarrolla este evento?",
    columna_motivo: str = "¿Cuál fue el motivo de su viaje a esta ciudad o municipio?"
) -> pd.Series:
    """
    Devuelve un Series con el conteo de categorías de motivo entre NO residentes.
    Sirve para poblar el selectbox en la UI.
    """
    if columna_reside not in df_encuesta.columns:
        raise ValueError(f"No se encontró la columna de residencia: '{columna_reside}'")

    if columna_motivo not in df_encuesta.columns:
        raise ValueError(f"No se encontró la columna de motivo: '{columna_motivo}'")

    # Filtrar respuestas válidas y NO residentes
    df_responde = df_encuesta[
        df_encuesta[columna_reside]
        .astype(str).str.strip().str.lower()
        .isin(["sí", "si", "no"])
    ]
    no_reside = df_responde[
        df_responde[columna_reside]
        .astype(str).str.strip().str.lower()
        .eq("no")
    ]

    if no_reside.empty:
        return pd.Series(dtype="int64")

    motivos = (
        no_reside[columna_motivo]
        .astype(str)
        .str.strip()
        .replace({"": "sin respuesta"})
        .str.lower()
    )

    return motivos.value_counts(dropna=False)


def calcular_pnl(
    df_encuesta: pd.DataFrame,
    df_aforo: pd.DataFrame,
    columna_reside: str = "¿Reside en la ciudad donde se desarrolla este evento?",
    columna_motivo: str = "¿Cuál fue el motivo de su viaje a esta ciudad o municipio?",
    categoria_principal: str | None = None,
    peso_principal: float = 1.0,
    peso_otros: float = 0.5,
    activar_factor_correccion: bool = False,
    factor_pt_n_sobre_rho: float | None = None,   # <<< NUEVO
) -> dict:
    """
    Calcula el PNL permitiendo seleccionar qué categoría de motivo es la 'principal'
    para el ponderador. El resto de categorías toman 'peso_otros'.
    """
    # 1) Filtrado de respuestas válidas
    if columna_reside not in df_encuesta.columns:
        raise ValueError(f"No se encontró la columna de residencia: '{columna_reside}'")

    df_encuesta_responde = df_encuesta[
        df_encuesta[columna_reside]
        .astype(str).str.strip().str.lower()
        .isin(["sí", "si", "no"])
    ]
    total_encuestados = df_encuesta_responde.shape[0]
    if total_encuestados == 0:
        raise ValueError("No hay encuestados válidos (Sí/No) en la columna de residencia.")

    # 2) Potencial de aforo (suma de todos los eventos)
    if "Potencial de aforo" not in df_aforo.columns:
        raise ValueError("El archivo de Aforo debe tener la columna 'Potencial de aforo'.")
    potencial_aforo = pd.to_numeric(df_aforo["Potencial de aforo"], errors="coerce").fillna(0).sum()

    # 3) NO residentes
    no_reside = df_encuesta_responde[
        df_encuesta_responde[columna_reside]
        .astype(str).str.strip().str.lower()
        .eq("no")
    ]
    total_no_reside = no_reside.shape[0]
    if total_no_reside == 0:
        return {
            "PNL": 0.0,
            "total_encuestados": total_encuestados,
            "potencial_aforo": potencial_aforo,
            "total_no_reside": 0,
            "total_motivo_seleccionado": 0,
            "proporcion_turismo": 0.0,
            "ponderador": 0.0,
            "no_reside": no_reside,
            "categoria_principal": categoria_principal,
            "peso_principal": peso_principal,
            "peso_otros": peso_otros,
            # Trazabilidad PT aunque sea 0
            "PT_con_repeticion": 0.0,
            "PT_ajustado": 0.0,
            "factor_pt_n_sobre_rho": None,
            "correccion_activada": False,
        }

    # 4) Homogeneizar motivo
    if columna_motivo not in df_encuesta.columns:
        raise ValueError(f"No se encontró la columna de motivo: '{columna_motivo}'")

    motivos_norm = (
        no_reside[columna_motivo]
        .astype(str)
        .str.strip()
        .replace({"": "sin respuesta"})
        .str.lower()
    )

    # 5) Selección de categoría principal
    if categoria_principal is None:
        vc = motivos_norm.value_counts(dropna=False)
        if "venir a los eventos religiosos" in vc.index:
            categoria_principal = "venir a los eventos religiosos"
        elif not vc.empty:
            categoria_principal = vc.idxmax()
        else:
            categoria_principal = "sin respuesta"

    # 6) Conteos
    total_motivo_sel = (motivos_norm == categoria_principal).sum()
    total_otras = total_no_reside - total_motivo_sel

    # 7) Proporción turismo
    proporcion_turismo = total_no_reside / total_encuestados

    # 8) Ponderador base (NO se corrige nunca con n/ρ)
    frac_principal = (total_motivo_sel / total_no_reside) if total_no_reside else 0.0
    frac_otras = ((total_no_reside - total_motivo_sel) / total_no_reside) if total_no_reside else 0.0
    num_categorias = motivos_norm.nunique(dropna=False)

    ponderador = (peso_principal * frac_principal) + (peso_otros * frac_otras)
    peso_principal_efectivo = peso_principal

    # 9) PT con repetición (según metodología)
    PT_con_repeticion = float(potencial_aforo) * float(proporcion_turismo)

    # 10) Aplicar corrección PT̃ = (n/ρ)·PT si corresponde
    if activar_factor_correccion and (factor_pt_n_sobre_rho is not None):
        f = max(0.0, min(1.0, float(factor_pt_n_sobre_rho)))
        PT_ajustado = f * PT_con_repeticion
        correccion_pt_activada = True
    else:
        PT_ajustado = PT_con_repeticion
        f = None
        correccion_pt_activada = False

    # 11) PNL final
    PNL = PT_ajustado * float(ponderador)


    return {
        "PNL": float(PNL),
        "total_encuestados": int(total_encuestados),
        "potencial_aforo": float(potencial_aforo),
        "total_no_reside": int(total_no_reside),
        "total_motivo_seleccionado": int(total_motivo_sel),
        "proporcion_turismo": float(proporcion_turismo),
        "ponderador": float(ponderador),
        "no_reside": no_reside,
        "categoria_principal": categoria_principal,
        "peso_principal": float(peso_principal),
        "peso_otros": float(peso_otros),
        "peso_principal_efectivo": float(peso_principal_efectivo),
        "num_categorias_motivo": int(num_categorias),
        "factor_correccion_aplicado": float(frac_otras),  # (total_otras/total_no_reside) — del ponderador
        "correccion_activada": bool(correccion_pt_activada),
        # >>> Trazabilidad PT
        "factor_pt_n_sobre_rho": (float(f) if f is not None else None),
    }



def evaluar_distribuciones(df, columnas, criterio="auto"):
    """
    Evalúa si las columnas seleccionadas tienen distribución normal.

    Parámetros:
        df: DataFrame
        columnas: Lista de nombres de columnas numéricas
        criterio: 'auto', 'Mediana' o 'Promedio'

    Retorna:
        dict con estadísticas (p-value, media, mediana, sugerencia)
    """
    resultados = {}
    for col in columnas:
        datos = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(datos) < 3:
            resultados[col] = {
                "N": len(datos),
                "p_value": np.nan,
                "media": np.nan,
                "mediana": np.nan,
                "sugerencia": "Insuficiente"
            }
            continue

        p_valor = sci_stats.shapiro(datos)[1]
        sugerencia = (
            "Promedio" if (criterio == "auto" and p_valor > 0.05) else "Mediana"
        ) if criterio == "auto" else criterio

        resultados[col] = {
            "N": len(datos),
            "p_value": p_valor,
            "media": datos.mean(),
            "mediana": datos.median(),
            "sugerencia": sugerencia
        }

    return resultados


def calcular_efecto_economico_indirecto(
    stats,
    pnl,
    multiplicador,
    col_aloj,
    col_alim,
    col_trans,
    col_dias,
    multiplicadores=None,
    extras=None,  # <<< NUEVO: lista de dicts [{"name": "...", "col": "...", "mult": float}, ...]
):
    """
    Calcula efectos por rubro usando los valores sugeridos de `stats`.
    Rubros base: alojamiento, alimentación, transporte.
    Extras: lista opcional de sectores adicionales con su columna y multiplicador.

    Para cada rubro r:
        Indirecto_r    = PNL * (valor_sugerido_r) * (dias_sugerido)
        InducidoNeto_r = (Indirecto_r * m_r) - Indirecto_r
    """
    def _num(x):
        try: return float(x)
        except (TypeError, ValueError): return float("nan")

    def _valor(col):
        sug = stats[col]["sugerencia"]
        return _num(stats[col]["media"] if sug == "Promedio" else stats[col]["mediana"])

    # Valores sugeridos desde stats
    v_aloj = _valor(col_aloj); v_alim = _valor(col_alim); v_trans = _valor(col_trans); dias = _valor(col_dias)
    v_aloj0 = 0.0 if pd.isna(v_aloj) else v_aloj
    v_alim0 = 0.0 if pd.isna(v_alim) else v_alim
    v_trans0 = 0.0 if pd.isna(v_trans) else v_trans
    dias0 = 0.0 if pd.isna(dias) else dias

    # Multiplicadores
    m_general = float(multiplicador)
    multiplicadores = multiplicadores or {}
    m_aloj = float(multiplicadores.get("alojamiento", m_general))
    m_alim = float(multiplicadores.get("alimentacion", m_general))
    m_trans = float(multiplicadores.get("transporte", m_general))

    pnl_f = float(pnl)

    # ---- Base: 3 rubros
    ind_aloj = pnl_f * v_aloj0 * dias0
    ind_alim = pnl_f * v_alim0 * dias0
    ind_trans = pnl_f * v_trans0 * dias0

    indirecto_total = ind_aloj + ind_alim + ind_trans
    inc_aloj = (ind_aloj * m_aloj) - ind_aloj
    inc_alim = (ind_alim * m_alim) - ind_alim
    inc_trans = (ind_trans * m_trans) - ind_trans
    inducido_neto_total = inc_aloj + inc_alim + inc_trans

    desglose = [
        {"Rubro": "Alojamiento", "Gasto diario usado": v_aloj0, "Indirecto": ind_aloj, "Inducido neto": inc_aloj},
        {"Rubro": "Alimentación", "Gasto diario usado": v_alim0, "Indirecto": ind_alim, "Inducido neto": inc_alim},
        {"Rubro": "Transporte",  "Gasto diario usado": v_trans0, "Indirecto": ind_trans, "Inducido neto": inc_trans},
    ]

    # ---- Extras dinámicos
    extras = extras or []
    mult_extras_dict = {}  # para devolver trazabilidad de multiplicadores de extras
    for ex in extras:
        name = str(ex.get("name", "Sector extra")).strip()
        col  = ex.get("col")
        m_ex = float(ex.get("mult", m_general))
        if not col or col not in stats:
            # columna inválida: registra en 0 para no romper cálculo
            v_ex0 = 0.0
        else:
            v_ex = _valor(col)
            v_ex0 = 0.0 if pd.isna(v_ex) else v_ex

        ind_ex = pnl_f * v_ex0 * dias0
        inc_ex = (ind_ex * m_ex) - ind_ex

        indirecto_total += ind_ex
        inducido_neto_total += inc_ex
        desglose.append({"Rubro": name, "Gasto diario usado": v_ex0, "Indirecto": ind_ex, "Inducido neto": inc_ex})
        mult_extras_dict[name] = m_ex

    # Fila total
    total_gasto_diario_usado = v_aloj0 + v_alim0 + v_trans0 + sum(
        0.0 if pd.isna(_valor(ex["col"])) else (_valor(ex["col"])) for ex in extras if ex.get("col") in (stats or {})
    )
    desglose.append({
        "Rubro": "Total",
        "Gasto diario usado": total_gasto_diario_usado,
        "Indirecto": indirecto_total,
        "Inducido neto": inducido_neto_total
    })

    resultado = {
        "PNL": pnl_f,
        "Días de estadía (valor usado)": dias0,
        "Multiplicador general": m_general,
        "Multiplicador alojamiento": m_aloj,
        "Multiplicador alimentación": m_alim,
        "Multiplicador transporte": m_trans,
        "Multiplicadores extras": mult_extras_dict,  # <<< NUEVO
        "Efecto Indirecto Total": indirecto_total,
        "Efecto Inducido Neto Total": inducido_neto_total
    }
    return resultado, desglose

def calcular_desglose_por_sectores(
    df_eed: pd.DataFrame,
    pnl: float,
    dias_usado: float,
    col_sector: str = "Sector_EED",
    col_valor: str = "V_EED",
    config_sectores: list | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Construye una tabla sectorial a partir del EED:
      - 'Efecto directo' = suma de V_EED por Sector_EED.
      - Opcional: 'Efecto indirecto' = PNL * gasto_sector * dias_usado (si activar=True).
      - 'Total, efecto inducido neto' = inducido(directo) + inducido(indirecto)
           donde inducido(x) = (x * multiplicador) - x.

    Retorna:
      - df_resultado con columnas:
          ['Sector','Efecto directo','Efecto indirecto',
           'Total, efecto inducido neto','Efecto económico total','% efecto económico total']
      - meta: dict con trazabilidad.
    """
    if col_sector not in df_eed.columns or col_valor not in df_eed.columns:
        raise ValueError(f"EED debe tener columnas '{col_sector}' y '{col_valor}'.")

    # Agregación por sector: efecto directo
    df_base = (
        df_eed[[col_sector, col_valor]]
        .assign(**{col_valor: pd.to_numeric(df_eed[col_valor], errors="coerce")})
        .groupby(col_sector, dropna=False, as_index=False)
        .sum()
        .rename(columns={col_sector: "Sector", col_valor: "Efecto directo"})
    )

    # Config default por sector
    cfg_map = {}
    for _, row in df_base.iterrows():
        nombre = str(row["Sector"])
        cfg_map[nombre] = {"activar": False, "gasto": 0.0, "multiplicador": 1.0}

    # Sobrescribir con configuración provista por la UI
    if config_sectores:
        for c in config_sectores:
            nombre = str(c.get("sector", ""))
            if nombre in cfg_map:
                cfg_map[nombre]["activar"] = bool(c.get("activar", False))
                cfg_map[nombre]["gasto"] = float(c.get("gasto", 0.0))
                cfg_map[nombre]["multiplicador"] = float(c.get("multiplicador", 1.0))

    efectos_indirecto = []
    efectos_inducido = []
    trazas = {}

    # Cálculos por sector
    for _, row in df_base.iterrows():
        nombre = str(row["Sector"])
        directo = float(row["Efecto directo"]) if pd.notna(row["Efecto directo"]) else 0.0
        cfg = cfg_map[nombre]
        m = float(cfg["multiplicador"])

        # Indirecto (opcional)
        if cfg["activar"]:
            indirecto = float(pnl) * float(cfg["gasto"]) * float(dias_usado)
        else:
            indirecto = 0.0

        # NUEVA FÓRMULA: inducido neto = inducido(directo) + inducido(indirecto)
        inc_directo = (directo * m) - directo
        inc_indirecto = (indirecto * m) - indirecto
        inducido_neto = inc_directo + inc_indirecto

        efectos_indirecto.append(indirecto)
        efectos_inducido.append(inducido_neto)

        trazas[nombre] = {
            "usar_indirecto": cfg["activar"],
            "gasto_sector": float(cfg["gasto"]),
            "multiplicador_sector": m,
            "inducido_directo": inc_directo,
            "inducido_indirecto": inc_indirecto,
        }

    # Construcción de la tabla
    df_res = df_base.copy()
    df_res["Efecto indirecto"] = efectos_indirecto
    df_res["Total, efecto inducido neto"] = efectos_inducido

    # Efecto económico total y participación
    df_res["Efecto económico total"] = (
        df_res["Efecto directo"]
        + df_res["Efecto indirecto"]
        + df_res["Total, efecto inducido neto"]
    )
    total_eco = float(df_res["Efecto económico total"].sum())
    if total_eco > 0:
        df_res["% efecto económico total"] = df_res["Efecto económico total"] / total_eco
    else:
        df_res["% efecto económico total"] = 0.0

    # Fila Total
    fila_total = pd.DataFrame({
        "Sector": ["Total"],
        "Efecto directo": [df_res["Efecto directo"].sum()],
        "Efecto indirecto": [df_res["Efecto indirecto"].sum()],
        "Total, efecto inducido neto": [df_res["Total, efecto inducido neto"].sum()],
        "Efecto económico total": [df_res["Efecto económico total"].sum()],
        "% efecto económico total": [1.0],
    })
    df_res = pd.concat([df_res, fila_total], ignore_index=True)

    # Orden de columnas
    cols = [
        "Sector",
        "Efecto directo",
        "Efecto indirecto",
        "Total, efecto inducido neto",
        "Efecto económico total",
        "% efecto económico total",
    ]
    df_res = df_res[cols]

    meta = {
        "PNL_usado": float(pnl),
        "dias_usado": float(dias_usado),
        "config_aplicada": trazas,
        "total_efecto_economico": float(total_eco),
    }
    return df_res, meta


