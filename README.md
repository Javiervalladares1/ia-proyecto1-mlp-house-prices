# Proyecto 1 - Competencia de modelacion con MLP

Implementacion reproducible de un Multi-Layer Perceptron para predecir `SalePrice` en el dataset Ames Housing. El criterio oficial es RMSE en la escala original del precio. El modelo de competencia es exclusivamente un MLP; Ridge se utiliza solo como referencia interna.

## Resultado principal

- CV confirmatoria estratificada, 5 folds: **26,847.57 ± 3,545.49 USD**.
- Holdout interno intacto (20%, abierto una sola vez): **17,754.33 USD**.
- Baseline MLP: **36,873.49 USD** de RMSE CV.
- Arquitectura final: `512 -> 256 -> 128 -> 64 -> 1`, activacion GELU.
- Seed global: `42`.
- Repositorio: <https://github.com/Javiervalladares1/ia-proyecto1-mlp-house-prices>

El artefacto en `models/final/` fue reentrenado con las 1,168 observaciones después de seleccionar la configuracion exclusivamente con validacion cruzada. Por eso el holdout reportado proviene del artefacto separado `models/validation_model/`, no de una evaluacion contaminada del modelo final.

## Estructura

```text
.
├── data/raw/                 # train.csv local, ignorado por Git
├── src/                      # EDA, preprocessing y definicion del MLP
├── models/final/             # pesos, preprocesador, target y metadatos
├── artifacts/                # tablas EDA, metricas y analisis de errores
├── figures/                  # visualizaciones y curvas de entrenamiento
├── experiments/              # results.csv, trials Optuna e historiales
├── predictions/              # salidas locales, ignoradas por Git
├── reports/                  # informe editable y PDF final
├── tests/                    # pruebas del pipeline de competencia
├── run_eda.py
├── train.py
└── predict.py
```

## Entorno e instalacion

Se desarrollo en macOS ARM64 con Apple M1 Pro, 16 GB de memoria, Python 3.13.0 y PyTorch 2.13.0. MPS fue detectado, pero CPU resulto la opcion estable y apropiada para este dataset pequeno, evitando transferencia y overhead de kernels.

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

Experimentos, busqueda Optuna, CV, holdout y reentrenamiento final:

```bash
python train.py --trials 18
```

La busqueda usa una reserva interna estratificada del 20%, CV de 3 folds para exploracion/Optuna y una confirmacion independiente de 5 folds sobre tres candidatos. Imputadores, clipping, deteccion de asimetria, escaladores, codificadores y seleccion de categorias se ajustan dentro de cada fold.

Pruebas:

```bash
python -m unittest discover -s tests -v
```

## Prediccion del dataset held-out

El comando operativo para el dia de la competencia es:

```bash
python predict.py /ruta/al/test.csv
```

Esto carga automaticamente `models/final/`, valida columnas, reordena features, tolera categorias no vistas, aplica exactamente el preprocessing guardado e inserta `Id,SalePrice` en `predictions/predictions.csv`.

Para elegir otro nombre de salida:

```bash
python predict.py /ruta/al/test.csv --output predictions/heldout_predictions.csv
```

Si el CSV contiene `SalePrice`, el script tambien imprime el RMSE en escala original. No es necesario modificar codigo.

## Preprocessing final

- `MSSubClass` y `MoSold` tratados como categorias, no como magnitudes continuas.
- Variables categoricas: imputacion explicita `Missing`, agrupacion de niveles infrecuentes (`min_frequency=5`) y one-hot con `handle_unknown="ignore"`.
- Variables numericas: mediana + indicadores de ausencia, clipping entrenado en percentiles 1/99, `log1p` para columnas no negativas con sesgo absoluto mayor que 0.75 y `RobustScaler`.
- Features de dominio deterministas: superficies totales, banos equivalentes, porches, antiguedad al vender, tiempo desde remodelacion e interacciones de calidad/superficie.
- Objetivo final: `SalePrice` original, estandarizado durante optimizacion y desestandarizado para toda metrica/prediccion.

## Artefactos importantes

- `experiments/results.csv`: iteraciones completas con metricas reales.
- `experiments/optuna_trials.csv`: 18 trials, incluidos los podados.
- `artifacts/best_configuration.json`: configuracion y metricas seleccionadas.
- `artifacts/holdout_metrics.json`: evaluacion honesta del holdout.
- `models/final/metadata.json`: columnas, versiones, hardware e hiperparametros.
- `reports/Informe_Proyecto_1_MLP.pdf`: informe final.

## Reproducibilidad y leakage

La division de competencia simulada utiliza seed `20260817`; las demas fuentes aleatorias usan `42` y seeds derivadas por fold. Ninguna transformacion que aprende estadisticas se ajusta fuera de los datos de entrenamiento del fold. El holdout no interviene en la eleccion de arquitectura, preprocessing, hiperparametros ni numero de epocas.

