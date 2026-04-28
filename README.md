# Benchmark de Modelos de Visión Local - Proyecto Siali

Este repositorio tiene como objetivo evaluar diferentes modelos de visión de última generación (VLM y modelos de detección) para su ejecución en local. El enfoque principal es la detección multiclase aplicada a entornos industriales y de seguridad.

## 🎯 Objetivos del Proyecto

Evaluar la capacidad de modelos pre-entrenados para identificar:
* **EPIs** (Equipos de Protección Individual).
* **Carretillas elevadoras**.
* **Detección de caídas**.

El proyecto busca determinar la viabilidad técnica de estos modelos mediante el análisis de su precisión, recall y facilidad de implementación sin necesidad de re-entrenamiento inicial.

## 👥 Equipo
* Miembro 1 (@arieram-siali) - Responsable de Rama A
* Miembro 2 (@usuario2) - Responsable de Rama B
* Miembro 3 (@usuario3) - Responsable de Rama C

## 🛠️ Modelos bajo investigación

Se han seleccionado los siguientes candidatos para las pruebas (cada miembro elegirá uno para implementar):

### Modelos de Investigación/VLM:
* **CogAgent** (Requiere GPU de alto rendimiento / Torre 4).
* **MobileVLM** (Optimizado para dispositivos ligeros).
* **Ferret-UI Lite**.
* **NVIDIA DeepStream + PeopleNet**.
* **TRT_Pose** (NVIDIA).

### Modelos Productizables (Recomendados):
* **InternVL2-Llama3** (o versiones U).
* **Phi-4 Reasoning Vision**.
* **DeepSeek Janus-Pro / DeepSeek-V4**.

---

## 🚀 Flujo de Trabajo (Git Workflow)

Para simular un entorno de proyecto real y mantener el repositorio organizado, seguiremos unas reglas estrictas para nombrar ramas y commits.

### 🌿 Convención de Nomenclatura de Ramas

Una rama de Git debe seguir el siguiente patrón:
`git branch <categoría>/<referencia>/<descripción>`

**1. Categoría:** La rama debe comenzar con una de estas opciones:
* `feature`: para agregar, refactorizar o eliminar una característica.
* `bugfix`: para corregir un error.
* `hotfix`: para cambiar el código con una solución temporal y/o sin seguir el proceso habitual (generalmente debido a una emergencia).
* `test`: para experimentar fuera de un problema/ticket.

**2. Referencia:**
Después de la categoría, debe haber una `/` seguida de la referencia del problema/ticket en el que se está trabajando. Si no hay referencia, simplemente agrega `no-ref`.

**3. Descripción:**
Después de la referencia, debe haber otra `/` seguida de una descripción que resuma el propósito de esta rama específica. Esta descripción debe ser corta y estar escrita en `kebab-case`. Simplemente reemplaza cualquier carácter especial o espacio por un `-`.

## 📋 Requisitos de Licencia
Se priorizarán modelos con licencias **Apache 2.0** o **MIT** para asegurar la viabilidad comercial del proyecto sin costes de licenciamiento.

---
*Este proyecto es parte de un ejercicio de evaluación técnica para el equipo de desarrollo de Siali.*

**Ejemplos:**
```bash
# Si necesita agregar, refactorizar o eliminar una característica:
git branch feature/issue-42/create-new-button-component

# Si necesita corregir un error:
git branch bugfix/issue-342/button-overlap-form-on-mobile

# Si necesita corregir un error muy rápido (posiblemente con una solución temporal):
git branch hotfix/no-ref/registration-form-not-working

# Si necesita experimentar fuera de un problema/ticket:
git branch test/no-ref/refactor-components-with-atomic-design
