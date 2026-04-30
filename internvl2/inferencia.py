"""
InternVL2-8B — Script de inferencia para benchmark de seguridad industrial
Detecta: EPIs (cascos, chalecos, guantes), carretillas, caídas de personas

Uso:
    python inferencia.py                        # procesa data/images/ completo
    python inferencia.py --imagen foto.jpg      # procesa una sola imagen
    python inferencia.py --sin-quant            # sin quantización (necesita más VRAM)
"""

import argparse
import json
import time
import warnings
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

warnings.filterwarnings("ignore")

# CONFIGURACIÓN DE RUTAS RELATIVAS 
# Detecta la carpeta raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent

# Configuración (Rellenar Path)

MODEL_ID = "OpenGVLab/InternVL2-8B"
DATA_DIR = BASE_DIR / "data" / "images"
RESULTS_DIR = BASE_DIR / "results"
IMAGENES_SOPORTADAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Asegurar que las carpetas existan
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Prompts con contexto industrial mínimo para reducir falsos positivos
PROMPTS = {
    "epi_casco": (
        "Detect all safety hard hats in this image. Count them and tell me the total number found. "
        'Reply only: {"casco": int}'
    ),
    "epi_chaleco": (
        "Check this image for safety vests. Count how many are and let me know the total."
        'Reply only: {"chaleco": int}'
    ),
    "epi_chaleco_negro": (
        "Detect black utility work vests only. Exclude any jackets or long-sleeved outerwear. Count the total number."
        'Reply only: {"chaleco_negro": int}'
    ),
    "epi_cono": (
        "Check this image for cones. Count all of them and let me know the total number visible."
        'Reply only: {"cono": int}'
    ),
    "epi_gorra_protectora": (
        "Identify all safety bump cap (protective work cap). Count them and distinguish them from regular caps. Report the total number of people wearing this protective headgear."
        'Reply only: {"gorra_protectora": int}'
    ),
    "epi_mascarilla": (
        "Detect and count all safety masks or respirators in this image."
        'Reply only: {"mascarilla": int}'
    ),
    "epi_persona": (
        "Detect and count all people present in this image."
        'Reply only: {"persona": int}'
    ),
    "carretillas": (
        "Detect and count all forklifts in this image. Provide the total number found."
        'Reply only: {"carretillas_detectadas": int}'
    ),
    "caidas": (
        "Analyze the image for fall detection. Identify if any individual appears to be in the middle of a fall, collapsing, or lying on the ground due to an accident."
        'Reply only: {"persona_caida": bool}'
    ),
}


# Carga del modelo

def cargar_modelo(usar_quantizacion: bool = True):
    """Carga InternVL2-8B con o sin quantización 4-bit."""
    print(f"\n{'='*60}")
    print(f"Cargando InternVL2-8B {'(4-bit quant)' if usar_quantizacion else '(BF16 completo)'}")
    print(f"{'='*60}")

    if usar_quantizacion:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModel.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto" 
        )
    else:
        model = AutoModel.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        use_fast=False,
    )

    model.eval()
    print("Modelo cargado correctamente.\n")
    return model, tokenizer


# Inferencia

def cargar_pixel_values(imagen_path: Path, model) -> torch.Tensor:
    """Preprocesa una imagen al formato que espera InternVL2 (pixel_values)."""
    from torchvision import transforms as T

    imagen = Image.open(imagen_path).convert("RGB")

    transform = T.Compose([
        T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    param = next(model.parameters())
    pixel_values = transform(imagen).unsqueeze(0).to(dtype=param.dtype, device=param.device)
    return pixel_values


def inferir(model, tokenizer, imagen_path: Path, prompt: str) -> dict:
    """Ejecuta inferencia sobre una imagen con el prompt dado."""
    pixel_values = cargar_pixel_values(imagen_path, model)

    generation_config = dict(
        max_new_tokens=512,
        do_sample=False,
    )

    t_inicio = time.time()
    respuesta = model.chat(
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        question=prompt,
        generation_config=generation_config,
        history=None,
        return_history=False,
    )
    latencia_ms = (time.time() - t_inicio) * 1000

    respuesta_json = None
    try:
        inicio = respuesta.find("{")
        fin = respuesta.rfind("}") + 1
        if inicio != -1 and fin > inicio:
            respuesta_json = json.loads(respuesta[inicio:fin])
    except json.JSONDecodeError:
        respuesta_json = None

    return {
        "respuesta_raw": respuesta,
        "respuesta_json": respuesta_json,
        "latencia_ms": round(latencia_ms, 1),
    }


# Procesamiento de imágenes

def procesar_imagen(model, tokenizer, imagen_path: Path) -> dict:
    """Ejecuta todos los prompts sobre una imagen y devuelve resultados combinados."""
    print(f"  Procesando: {imagen_path.name}")

    resultados_prompts = {}
    for nombre_prompt, texto_prompt in PROMPTS.items():
        print(f"    → {nombre_prompt}...", end=" ", flush=True)
        resultado = inferir(model, tokenizer, imagen_path, texto_prompt)
        resultados_prompts[nombre_prompt] = resultado
        print(f"{resultado['latencia_ms']} ms")

    return {
        "imagen": imagen_path.name,
        "ruta": str(imagen_path),
        "resultados": resultados_prompts,
    }


def procesar_directorio(model, tokenizer, directorio: Path) -> list:
    """Procesa todas las imágenes de un directorio."""
    imagenes = [
        p for p in sorted(directorio.iterdir())
        if p.suffix.lower() in IMAGENES_SOPORTADAS
    ]

    if not imagenes:
        print(f"No se encontraron imágenes en {directorio}")
        return []

    print(f"\nEncontradas {len(imagenes)} imágenes en {directorio}\n")

    resultados = []
    for i, imagen_path in enumerate(imagenes, 1):
        print(f"[{i}/{len(imagenes)}]")
        resultado = procesar_imagen(model, tokenizer, imagen_path)
        resultados.append(resultado)

    return resultados


# Guardado de resultados

_PROMPT_CLAVES = {
    "epi_casco": "casco",
    "epi_chaleco": "chaleco",
    "epi_chaleco_negro": "chaleco_negro",
    "epi_cono": "cono",
    "epi_gorra_protectora": "gorra_protectora",
    "epi_mascarilla": "mascarilla",
    "epi_persona": "persona",
    "carretillas": "carretillas_detectadas",
    "caidas": "persona_caida",
}


def _agregar_detecciones(resultados: list) -> dict:
    """Agrega todas las detecciones en un dict plano con totales."""
    totales = {clave: 0 for clave in _PROMPT_CLAVES.values()}

    for resultado in resultados:
        res = resultado["resultados"]
        for nombre_prompt, clave in _PROMPT_CLAVES.items():
            datos = res.get(nombre_prompt, {}).get("respuesta_json") or {}
            valor = datos.get(clave)
            if isinstance(valor, bool):
                totales[clave] += int(valor)
            elif isinstance(valor, (int, float)):
                totales[clave] += int(valor)

    return totales


def guardar_resultados(resultados: list, directorio_salida: Path):
    """Guarda los resultados en JSON y genera un resumen legible."""
    directorio_salida.mkdir(parents=True, exist_ok=True)

    ruta_json = directorio_salida / "resultados_completos.json"
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    resumen_latencias = {}
    for resultado in resultados:
        for nombre_prompt, datos in resultado["resultados"].items():
            if nombre_prompt not in resumen_latencias:
                resumen_latencias[nombre_prompt] = []
            resumen_latencias[nombre_prompt].append(datos["latencia_ms"])

    resumen = {
        "total_imagenes": len(resultados),
        "modelo": MODEL_ID,
        "latencias_promedio_ms": {
            nombre: round(sum(vals) / len(vals), 1)
            for nombre, vals in resumen_latencias.items()
        },
        "detecciones_agregadas": _agregar_detecciones(resultados),
    }

    ruta_resumen = directorio_salida / "resumen.json"
    with open(ruta_resumen, "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Resultados guardados en {directorio_salida}")
    print(f"  - resultados_completos.json ({len(resultados)} imágenes)")
    print(f"  - resumen.json")
    print(f"\nLatencias promedio:")
    for nombre, lat in resumen["latencias_promedio_ms"].items():
        print(f"  {nombre:30s}: {lat} ms")
    print(f"\nDetecciones totales:")
    for clase, total in resumen["detecciones_agregadas"].items():
        print(f"  {clase:25s}: {total}")
    print(f"{'='*60}\n")


# Main

def main():
    parser = argparse.ArgumentParser(description="InternVL2-8B — Benchmark seguridad industrial")
    parser.add_argument(
        "--imagen",
        type=str,
        default=None,
        help="Ruta a una imagen específica (si no se indica, procesa todo data/images/)",
    )
    parser.add_argument(
        "--sin-quant",
        action="store_true",
        help="Desactiva quantización 4-bit (necesita ~16GB VRAM)",
    )
    args = parser.parse_args()

    model, tokenizer = cargar_modelo(usar_quantizacion=not args.sin_quant)

    if args.imagen:
        imagen_path = Path(args.imagen)
        if not imagen_path.exists():
            print(f"Error: no se encuentra {imagen_path}")
            return
        resultados = [procesar_imagen(model, tokenizer, imagen_path)]
    else:
        resultados = procesar_directorio(model, tokenizer, DATA_DIR)

    if resultados:
        guardar_resultados(resultados, RESULTS_DIR)


if __name__ == "__main__":
    main()