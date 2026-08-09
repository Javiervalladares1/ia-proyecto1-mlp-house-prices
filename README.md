# Proyecto 1 - Competencia de modelación con MLP

Implementación reproducible de un Multi-Layer Perceptron para predecir `SalePrice` en el dataset Ames Housing. El criterio oficial es RMSE en la escala original del precio. El modelo de competencia es exclusivamente un MLP; Ridge se utiliza solo como referencia interna.

## Resultado principal

- CV confirmatoria estratificada, 5 folds: **26,847.57 ± 3,545.49 USD**.
- Holdout interno intacto (20%, abierto una sola vez): **17,754.33 USD**.
- Baseline MLP: **36,873.49 USD** de RMSE CV.
- Arquitectura final: `512 -> 256 -> 128 -> 64 -> 1`, activación GELU.
- Seed global: `42`.
- Repositorio: <https://github.com/Javiervalladares1/ia-proyecto1-mlp-house-prices>

El artefacto en `models/final/` fue reentrenado con las 1,168 observaciones después de seleccionar la configuración exclusivamente con válidación cruzada. Por eso el holdout reportado proviene del artefacto separado `models/válidation_model/`, no de una evaluación contaminada del modelo final.

## Estructura

```text
.
├── data/raw/                 # train.csv local, ignorado por Git
├── src/                      # EDA, preprocessing y definición del MLP
├── models/final/             # pesos, preprocesador, target y metadatos
├── artifacts/                # tablas EDA, métricas y análisis de errores
├── figures/                  # visualizaciones y curvas de entrenamiento
├── experiments/              # results.csv, trials Optuna e historiales
├── predictions/              # salidas locales, ignoradas por Git
├── reports/                  # informe editable y PDF final
├── tests/                    # pruebas del pipeline de competencia
├── run_eda.py
├── train.py
└── predict.py
```

## Entorno e instalación

Se desarrolló en macOS ARM64 con Apple M1 Pro, 16 GB de memoria, Python 3.13.0 y PyTorch 2.13.0. MPS fue detectado, pero CPU resultó la opción estable y apropiada para este dataset pequeño, evitando transferencia y overhead de kernels.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Coloque el archivo de entrenamiento en `data/raw/train.csv`. Debe incluir las 80 columnas predictoras, `Id` y `SalePrice`.

## Reproduccion

EDA y figuras:

```bash
python run_eda.py
```

Experimentos, búsqueda Optuna, CV, holdout y reentrenamiento final:

```bash
python train.py --trials 18
```

La búsqueda usa una reserva interna estratificada del 20%, CV de 3 folds para exploración/Optuna y una confirmación independiente de 5 folds sobre tres candidatos. Imputadores, clipping, deteccion de asimetría, escaladores, codificadores y selección de categorías se ajustan dentro de cada fold.

Pruebas:

```bash
python -m unittest discover -s tests -v
```

## Predicción del dataset held-out

El comando operativo para el dia de la competencia es:

```bash
python predict.py /ruta/al/test.csv
```

Esto carga automaticamente `models/final/`, válida columnas, reordena features, tolera categorías no vistas, aplica exactamente el preprocessing guardado e inserta `Id,SalePrice` en `predictions/predictions.csv`.

Para elegir otro nombre de salida:

```bash
python predict.py /ruta/al/test.csv --output predictions/heldout_predictions.csv
```

Si el CSV contiene `SalePrice`, el script también imprime el RMSE en escala original. No es necesario modificar código.

## Preprocessing final

- `MSSubClass` y `MoSold` tratados como categorías, no como magnitudes continuas.
- Variables categóricas: imputación explícita `Missing`, agrupación de niveles infrecuentes (`min_frequency=5`) y one-hot con `handle_unknown="ignore"`.
- Variables numéricas: mediana + indicadores de ausencia, clipping entrenado en percentiles 1/99, `log1p` para columnas no negativas con sesgo absoluto mayor que 0.75 y `RobustScaler`.
- Features de dominio deterministas: superficies totales, baños equivalentes, porches, antigüedad al vender, tiempo desde remodelacion e interacciones de calidad/superficie.
- Objetivo final: `SalePrice` original, estandarizado durante optimización y desestandarizado para toda métrica/predicción.

## Artefactos importantes

- `experiments/results.csv`: iteraciones completas con métricas reales.
- `experiments/optuna_trials.csv`: 17 trials bayesianos, incluidos 5 podados tempranamente.
- `artifacts/best_configuration.json`: configuración y métricas seleccionadas.
- `artifacts/holdout_metrics.json`: evaluación honesta del holdout.
- `models/final/metadata.json`: columnas, versiones, hardware e hiperparámetros.
- `reports/Informe_Proyecto_1_MLP.pdf`: informe final.

## Reproducibilidad y leakage

La división de competencia simulada utiliza seed `20260817`; las demas fuentes aleatorias usan `42` y seeds derivadas por fold. Ninguna transformación que aprende estadisticas se ajusta fuera de los datos de entrenamiento del fold. El holdout no interviene en la elección de arquitectura, preprocessing, hiperparámetros ni número de épocas.
