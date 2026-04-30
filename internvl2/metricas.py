import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# ── CONFIGURACIÓN (Extraída de tu script funcional) ──────────────────────────
PREFIJO_A_CLASE = {
    "caida":             "persona_caida",
    "carretilla":        "carretillas_detectadas",
    "casco":             "casco",
    "chaleco_negro":     "chaleco_negro",
    "chaleco":           "chaleco",
    "cono":              "cono",
    "gorra_protectora":  "gorra_protectora",
    "mascarilla":        "mascarilla",
    "persona":           "persona",
}

CLASES = [
    "persona_caida", "carretillas_detectadas", "casco", "chaleco",
    "chaleco_negro", "cono", "gorra_protectora", "mascarilla", "persona",
]

ETIQUETAS = [
    "Caída", "Carretilla", "Casco", "Chaleco",
    "Chaleco negro", "Cono", "Gorra protectora", "Mascarilla", "Persona",
]

PROMPT_A_CAMPO = {
    "epi_casco":            "casco",
    "epi_chaleco":          "chaleco",
    "epi_chaleco_negro":    "chaleco_negro",
    "epi_cono":             "cono",
    "epi_gorra_protectora": "gorra_protectora",
    "epi_mascarilla":       "mascarilla",
    "epi_persona":          "persona",
    "carretillas":          "carretillas_detectadas",
    "caidas":               "persona_caida",
}

# ── FUNCIONES DE EXTRACCIÓN (Adaptadas de tu script) ──────────────────────────

def get_prefix(nombre_archivo):
    nombre = nombre_archivo.lower().rsplit(".", 1)[0]
    # Priorizar prefijos largos (evita que 'chaleco_negro' se detecte como 'chaleco')
    for prefijo in sorted(PREFIJO_A_CLASE.keys(), key=len, reverse=True):
        if nombre.startswith(prefijo):
            return prefijo
    return None

def get_pred(entry, campo_objetivo):
    """Navega por el JSON buscando el campo específico."""
    for prompt_key, campo_json in PROMPT_A_CAMPO.items():
        if campo_json == campo_objetivo:
            # Acceso seguro a la estructura anidada
            rj = entry.get("resultados", {}).get(prompt_key, {}).get("respuesta_json") or {}
            val = rj.get(campo_objetivo)
            if val is None: return 0
            if isinstance(val, bool): return 1 if val else 0
            return 1 if int(val) > 0 else 0
    return 0

# ── PROCESAMIENTO ─────────────────────────────────────────────────────────────

json_path = Path("results/resultados_completos.json") # Cambia a resultados_completos.json si es necesario
output_dir = Path("results")

with open(json_path, encoding="utf-8") as f:
    data = json.load(f)

resultados_metricas = []

for clase_id, nombre_legible in zip(CLASES, ETIQUETAS):
    tp, fp, fn, tn = 0, 0, 0, 0
    
    for entry in data:
        prefijo = get_prefix(entry["imagen"])
        clase_positiva_real = PREFIJO_A_CLASE.get(prefijo)
        
        real = 1 if clase_id == clase_positiva_real else 0
        pred = get_pred(entry, clase_id)
        
        if real == 1 and pred == 1: tp += 1
        elif real == 0 and pred == 1: fp += 1
        elif real == 1 and pred == 0: fn += 1
        else: tn += 1

    # Cálculo de scores
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    
    resultados_metricas.append({
        "Etiqueta": nombre_legible, "P": p, "R": r, "F1": f1,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn
    })

df = pd.DataFrame(resultados_metricas)

# ── GENERACIÓN DE GRÁFICAS ───────────────────────────────────────────────────

# 1. Gráfica de Barras de Métricas (P, R, F1)
plt.figure(figsize=(12, 6))
df_melt = df.melt(id_vars="Etiqueta", value_vars=["P", "R", "F1"], var_name="Métrica", value_name="Valor")
sns.barplot(data=df_melt, x="Etiqueta", y="Valor", hue="Métrica", palette="viridis")
plt.title("Métricas de Seguridad Industrial por Clase", fontsize=14)
plt.ylim(0, 1.1)
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "barras_metricas_adaptado.png")

# 2. Matriz de Confusión Resumida (Heatmap de TP/FP/FN/TN)
plt.figure(figsize=(10, 8))
df_cm = df.set_index("Etiqueta")[["TP", "FP", "FN", "TN"]]
sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu", cbar_kws={'label': 'Cantidad'})
plt.title("Matriz Resumen de Aciertos y Errores", fontsize=14)
plt.tight_layout()
plt.savefig(output_dir / "matriz_resumen_adaptado.png")

print(f"Proceso finalizado. Gráficas guardadas en {output_dir}")
print(df[["Etiqueta", "P", "R", "F1", "TP", "FP"]].to_string(index=False))