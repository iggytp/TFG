# TFG - Clasificador Estelar Gaia DR3

Proyecto de Trabajo de Fin de Grado para la clasificación estelar con datos de Gaia DR3. El repositorio reúne todo el flujo de trabajo: extracción y preparación de datos, exploración y entrenamiento en notebook, y una aplicación web interactiva para probar el modelo final.

## Contenido del proyecto

- `ETL.py`: descarga datos desde Gaia DR3, aplica filtros de calidad y genera `dataset_v1.csv`.
- `TFG.ipynb`: notebook principal del proyecto, con el análisis exploratorio, el preprocesado, el entrenamiento y la evaluación de modelos, y la generación del artefacto final.
- `webapp/app.py`: aplicación Streamlit para introducir variables predictoras, lanzar predicciones y analizar sensibilidad.
- `modelo_clasificacion_estelar.joblib`: modelo entrenado que consume la app.
- `dataset_v1.csv` y `dataset_predictoras_v1.csv`: datasets de referencia usados por el proyecto y por la interfaz.

## Estructura

```text
.
├── ETL.py
├── TFG.ipynb
├── modelo_clasificacion_estelar.joblib
├── requirements.txt
├── dataset_v1.csv
├── dataset_predictoras_v1.csv
├── webapp/
│   └── app.py
└── README.md
```

## Requisitos

- Python 3.10 o superior.
- Dependencias instaladas desde `requirements.txt`.
- Acceso a Internet para ejecutar el ETL contra el servidor TAP de Gaia.

## Instalacion

Desde la raiz del proyecto:

```bash
pip install -r requirements.txt
```

## Flujo de trabajo

### 1. Obtener el dataset

Ejecuta `ETL.py` para descargar una muestra balanceada de Gaia DR3 y generar `dataset_v1.csv`.

```bash
python ETL.py
```

El script conecta con el servidor TAP de la ESA, aplica filtros de calidad y selecciona objetos con alta probabilidad de ser estrella. A partir de los datos descargados calcula variables derivadas como errores fotometricos y magnitud absoluta.

### 2. Trabajar en el notebook

Abre `TFG.ipynb` para reproducir el proceso de analisis y modelado.

En el notebook se realiza, de forma general:

- carga y limpieza de datos;
- analisis exploratorio;
- separacion de variables predictoras y objetivo;
- escalado y transformaciones auxiliares;
- entrenamiento y comparacion de varios modelos;
- evaluacion de metricas y visualizaciones;
- guardado del modelo final en `modelo_clasificacion_estelar.joblib`.

### 3. Lanzar la app web

La aplicacion Streamlit permite probar el modelo con controles interactivos.

```bash
streamlit run webapp/app.py
```

La app carga automaticamente el modelo y toma como referencia `dataset_v1.csv` o `dataset_predictoras_v1.csv` si estan disponibles.

## Funcionalidades de la app

- Prediccion en tiempo real a partir de variables fotometricas.
- Probabilidades por clase, si el estimador las soporta.
- Comparativa del valor introducido frente al universo de referencia.
- Analisis `what-if` para ver como cambia la salida al variar una variable.
- Ficha tecnica del modelo con las variables usadas y los hiperparametros guardados.

## Nota sobre Gaia DR3

El proyecto usa la columna `classprob_dsc_combmod_star` de Gaia DR3 para filtrar objetos con alta probabilidad de ser estrella. En las consultas del ETL se aplica el criterio `classprob_dsc_combmod_star >= 0.95`.

## Datos y artefactos generados

Los archivos que normalmente debe generar o conservar el proyecto son:

- `dataset_v1.csv`
- `dataset_predictoras_v1.csv`
- `modelo_clasificacion_estelar.joblib`

Si alguno falta, ejecuta primero el ETL y despues el notebook para regenerar el modelo.

## Uso recomendado para demo

1. Ejecutar `ETL.py` si hace falta regenerar datos.
2. Abrir `TFG.ipynb` para revisar el entrenamiento y el contexto del modelo.
3. Lanzar la web con `streamlit run webapp/app.py`.
4. Probar cambios en `phot_rp_mean_mag`, `e_Gmag` y `bp_rp` para mostrar sensibilidad del modelo.


