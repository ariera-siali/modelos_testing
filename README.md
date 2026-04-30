**SIALI TECHNOLOGIES · Abril 2026**

**Informe de Benchmark**

Modelo InternVL2-8B — Detección de Seguridad Industrial

Alejandro Riera · Rama: feature/internvl2-8b

# **1\. Resumen Ejecutivo**

Se ha evaluado el modelo InternVL2-8B (MIT License, Shanghai AI Lab) sobre imágenes de entornos industriales para las categorías de EPIs, carretillas elevadoras y detección de caídas. El modelo se ejecuta en local sobre una NVIDIA RTX 4070 8GB con quantización 4-bit (NF4), sin fine-tuning ni entrenamiento adicional.

La conclusión principal es que InternVL2-8B funciona de forma notablemente fiable en tareas de detección binaria o de objetos grandes y contextualmente únicos (carretillas, caídas), pero muestra limitaciones claras en la discriminación de EPIs visualualmente similares o de pequeño tamaño, especialmente en imágenes con destellos o iluminación compleja.

# **2\. Configuración del Experimento**

## **Hardware y modelo**

| **GPU** | NVIDIA RTX 4070 Laptop — 8GB VRAM |
| --- | --- |
| **SO** | Ubuntu 24.04 LTS |
| --- | --- |
| **Modelo** | OpenGVLab/InternVL2-8B (MIT License) |
| --- | --- |
| **Quantización** | 4-bit NF4 (bitsandbytes) — BFloat16 compute |
| --- | --- |
| **Framework** | Transformers 4.40.2 + Accelerate + Docker |
| --- | --- |
| **Imágenes** | Imágenes reales de entorno industrial Siali |
| --- | --- |

## **Prompts utilizados**

Se utilizaron 9 prompts independientes por imagen, uno por clase, con instrucciones negativas explícitas para reducir falsos positivos:

- epi_casco — Detect all safety hard hats in this image. Count them and tell me the total number found.
- epi_chaleco — Check this image for safety vests. Count how many are and let me know the total.
- epi_chaleco_negro — Detect black utility work vests only. Exclude any jackets or long-sleeved outerwear. Count the total number.
- epi_cono — Check this image for cones. Count all of them and let me know the total number visible.
- epi_gorra_protectora — Identify all safety bump cap (protective work cap). Count them and distinguish them from regular caps. Report the total number of people wearing this protective headgear.
- epi_mascarilla — Detect and count all safety masks or respirators in this image.
- epi_persona — Detect and count all people present in this image.
- carretillas — Detect and count all forklifts in this image. Provide the total number found.
- caidas — Analyze the image for fall detection. Identify if any individual appears to be in the middle of a fall, collapsing, or lying on the ground due to an accident.

# **3\. Resultados de Latencia**

Latencias medidas sobre una RTX 4070 con quantización 4-bit, modelo ya cargado en VRAM. No incluye el tiempo de carga del modelo (~6 segundos adicionales en el primer arranque).

| **Prompt** | **Latencia promedio** | **Observación** |
| --- | --- | --- |
| epi_casco | 764 ms | Rápido y consistente |
| --- | --- | --- |
| epi_chaleco | 795.7 ms |     |
| --- | --- | --- |
| epi_chaleco_negro | 899.8 ms |     |
| --- | --- | --- |
| epi_cono | 746.7 ms | Más rápido del grupo EPI |
| --- | --- | --- |
| epi_gorra_protectora | 870.4 ms |     |
| --- | --- | --- |
| epi_mascarilla | 808.8 ms | Más lento |
| --- | --- | --- |
| epi_persona | 753 ms | Más rápido — clase más reconocible |
| --- | --- | --- |
| carretillas | 949.4 ms | Buena precisión |
| --- | --- | --- |
| caidas | 782.7 ms | Buena precisión |
| --- | --- | --- |

El tiempo total por imagen con los 9 prompts es de aproximadamente 7,37 segundos. Para un sistema de monitorización en tiempo real esto es inviable, pero para análisis offline de imágenes puntuales el rendimiento es aceptable.

# **4\. Análisis Cualitativo por Categoría**

## **4.1 Carretillas elevadoras — Rendimiento BUENO**

La detección de carretillas es fiable. Al ser un objeto de gran tamaño, forma característica y contexto industrial claro, el modelo lo identifica con pocas confusiones. Los falsos positivos observados son escasos y suelen ocurrir en imágenes con vehículos parcialmente visibles o con perspectivas inusuales.

## **4.2 Detección de caídas — Rendimiento BUENO**

La detección de personas en postura caída funciona sorprendentemente bien para un modelo sin fine-tuning. Al tratarse de una pregunta binaria (sí/no) con contexto visual claro, el modelo raramente genera falsos positivos cuando las personas están de pie o sentadas. La limitación principal aparece en posturas agachadas o en cuclillas que pueden confundirse con una caída.

## **4.3 Personas (epi_persona) — Rendimiento ACEPTABLE**

El conteo de personas es razonablemente preciso en escenas con pocas personas y buena visibilidad. Los problemas aparecen cuando hay solapamiento entre personas, oclusiones parciales o grupos densos. Se observan tanto falsos negativos (personas no contadas) como algún doble conteo.

## **4.4 Casco y chaleco reflectante — Rendimiento MODERADO**

Estos dos EPIs son los mejor detectados del grupo por su distintividad visual. El casco es un objeto de forma y posición bien definidos. El chaleco reflectante tiene colores y patrones llamativos. Aun así, aparecen confusiones en imágenes con iluminación artificial intensa o destellos que alteran la percepción de colores.

## **4.5 Chaleco negro, gorra protectora y mascarilla — Rendimiento DÉBIL**

Estas tres clases presentan el mayor número de falsos positivos. Las causas identificadas son:

- **Chaleco negro:** El modelo tiende a clasificar cualquier prenda oscura como chaleco de trabajo, incluyendo chaquetas, sudaderas y uniformes. La distinción entre ropa casual oscura y workwear no es fiable.
- **Gorra protectora:** Alta confusión con gorras deportivas, cascos y cabello oscuro con volumen. Al ser un objeto pequeño y de formas variadas, el modelo no tiene suficiente contexto visual para discriminar.
- **Mascarilla:** Elevados falsos positivos en imágenes con destellos o zonas sobreexpuestas cerca del área facial. El modelo interpreta reflejos o zonas brillantes como mascarillas.

# **5\. Limitaciones Identificadas**

## **5.1 Sensibilidad a destellos e iluminación**

Las imágenes de entornos industriales con iluminación artificial intensa (lámparas de techo, reflejos en superficies metálicas, sobreexposición) degradan significativamente la precisión en clases pequeñas. El modelo interpreta alteraciones del color/brillo como objetos de la clase buscada.

## **5.2 Ausencia de fine-tuning**

InternVL2-8B es un modelo de propósito general. Sin entrenamiento específico sobre imágenes industriales etiquetadas, su capacidad de discriminación fina entre clases visualmente similares es limitada. El benchmark demuestra que sirve como prueba de concepto pero no como solución de producción directa.

## **5.3 Quantización 4-bit**

La quantización NF4 necesaria para caber en 8GB VRAM introduce pérdida de precisión numérica que afecta especialmente a tareas de discriminación fina. Con el modelo en BF16 completo (16GB VRAM) los resultados mejorarían, pero esto requeriría hardware diferente.

## **5.4 Prompts independientes por clase**

La estrategia de un prompt por clase aumenta la latencia total y no aprovecha el contexto global de la imagen. Un único prompt que pida todas las clases a la vez podría ser más coherente, pero experimentalmente se demostró que genera más confusión cuando hay muchas clases simultáneas.

# **6\. Conclusiones y Recomendaciones**

InternVL2-8B demuestra ser un modelo viable para detección industrial de objetos grandes y contextualmente únicos sin necesidad de entrenamiento adicional. Su principal valor es la facilidad de despliegue (HuggingFace, MIT License, código de ejemplo disponible) y la velocidad de integración en un pipeline de prueba.

| **Categoría** | **Valoración** | **Recomendación** |
| --- | --- | --- |
| Carretillas | ✓ Buena | Apto para uso directo como alerta |
| --- | --- | --- |
| Caídas | ✓ Buena | Apto como primera capa de detección |
| --- | --- | --- |
| Personas | ~ Aceptable | Válido para conteo aproximado |
| --- | --- | --- |
| Casco / Chaleco | ~ Moderada | Requiere fine-tuning para producción |
| --- | --- | --- |
| Chaleco negro | ✗ Débil | No recomendado sin fine-tuning |
| --- | --- | --- |
| Gorra / Mascarilla | ✗ Débil | No recomendado — demasiados FP |
| --- | --- | --- |

Para un sistema de producción en Siali se recomienda combinar InternVL2-8B (o un modelo similar) con un detector clásico tipo YOLOv8 fine-tuned sobre datos propios: YOLO para las detecciones críticas con alta precisión, y el VLM como capa de razonamiento contextual para alertas complejas.

Alejandro Riera — Siali Technologies · Abril 2026
