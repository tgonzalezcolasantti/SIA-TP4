# SIA-TP4

Infraestructura de experimentación adaptada del TP3 propio: configuración y carga
de `ej2/utils.py`, barridos y repeticiones de `ej2/compare_activations.py`, y gráficos
separados del entrenamiento como en `ej2/plot_comparisons.py`.

## Instalación y ejecución

Requiere Python 3.14 y [uv](https://docs.astral.sh/uv/). Desde la raíz del repositorio:

```powershell
uv sync --locked
uv run python -m experiments.run_kohonen --config configs/kohonen.json
```

El comando carga los creadores, compara las configuraciones, guarda los resultados
y genera gráficos sin abrir ventanas. Las rutas `dataset` y `out_dir` del JSON se
resuelven respecto del archivo de configuración. Se puede usar un JSON alternativo.

Para regenerar solamente los gráficos:

```powershell
uv run python -m experiments.plot_results --results-dir results/kohonen
```

## Configuración

`configs/kohonen.json` define el producto cartesiano de tamaños de grilla, radios,
tasas iniciales y épocas. Cada combinación se repite con todas las semillas.
El ejemplo ejecuta 12 configuraciones con 3 semillas: 36 entrenamientos.

- `features`: lista explícita de variables; `null` usa todas salvo identificadores
  y `exclude_columns`.
- `exclude_columns`: variables a descartar cuando `features` es `null`.
- `seeds`: semillas distintas para inicialización, enteros entre 0 y 2**32 - 1.
- `grid_sizes`: lados de grillas cuadradas, enteros >= 2.
- `radii`: radios constantes, positivos y <= al menor tamaño de grilla del barrido.
- `learning_rates`: tasas constantes en (0, 1], como en el modelo original.
- `epochs`: épocas positivas; en cada época se recorre el dataset en el orden original.

`creator_id` y `name` se conservan para identificar resultados y nunca son entradas
del modelo. Las variables se estandarizan mediante z-score con desvío poblacional.
Las constantes quedan en cero y se registran. Los valores faltantes o infinitos
generan un error: no se imputan silenciosamente.

La selección inicial de todas las variables es una base para explorar, no una
justificación final. El EDA debe considerar redundancias como USD/ARS, porcentajes
complementarios y variables derivadas. Ajustar `features` o `exclude_columns` según
ese análisis y conservar la configuración de cada estudio en una carpeta diferente.

## Resultados

En `results/kohonen/` se generan:

| Archivo | Contenido |
| --- | --- |
| `config_used.json`, `metadata.json` | Configuración efectiva, hash del CSV, versiones y criterio de selección |
| `preprocessing.json` | Orden de variables, medias, escalas y constantes |
| `standardized_creators.csv` | Entradas utilizadas, con identificadores |
| `descriptive_statistics.csv`, `correlations.csv` | Estadísticas descriptivas y correlaciones para explorar los datos |
| `all_trials.csv` | Una fila por configuración y semilla, con métricas y duración |
| `summary.csv` | Media y desvío muestral entre semillas (cero si hay una sola) |
| `selection.json` | Configuración y semilla de la corrida representativa exportada |
| `selected_model.npz` | Pesos, tamaño de grilla, variables y parámetros de escalado |
| `assignments.csv` | Neurona y coordenadas de cada creador |
| `neuron_profiles.csv` | Media de las variables originales por neurona ocupada |
| `u_matrix.csv` | Distancias promedio a vecinos ortogonales |
| `plots/` | Comparación de errores, asignaciones, matriz U, planos de variables y correlaciones |

Las métricas son error de cuantización (distancia media al peso ganador), error
topográfico (proporción cuyos dos ganadores no son vecinos ortogonales), cantidad
de neuronas vacías y proporción de neuronas ocupadas. No se necesitan etiquetas.
Los empates de distancias se resuelven por índice de neurona.

Se exporta la configuración de menor error medio de cuantización, desempatando por
error topográfico. Dentro de ella se usa la semilla cuyo error está más cerca de la
media. Este criterio es una referencia técnica: grillas más grandes suelen reducir
la cuantización, por lo que hay que interpretar también topología, ocupación y
utilidad para las campañas antes de justificar la elección final.

La matriz U y el error topográfico usan cuatro vecinos ortogonales. El pipeline usa
`Kohonnen` sin modificar `models/unsupervised.py`: radio y tasa constantes, orden
de muestras fijo y entrenamiento al construir el objeto. La semilla se controla
desde el pipeline, que restaura el estado aleatorio global al terminar.

El modelo original excluye a la ganadora de la actualización. Esta limitación sigue
pendiente de corregir en el modelo y debe tenerse en cuenta al interpretar métricas.

`experiments.utils.transform_features` permite transformar nuevos creadores con
los parámetros de `preprocessing.json`, sin reajustar el escalado. Los mismos datos,
configuración, semillas y entorno reproducen los resultados; las duraciones varían.
Volver a ejecutar en el mismo `out_dir` reemplaza las salidas del estudio anterior.

## Verificación

```powershell
uv run python -m unittest discover -s tests -v
```

Incluye casos conocidos para métricas, constantes,
datos faltantes, transformación de nuevos datos y reproducción de un estudio completo.

## Alcance

Esta implementación conecta el Kohonen existente con los datos y los experimentos.
Los estadísticos y gráficos son insumos para el EDA; falta su interpretación y la
justificación de variables. PCA, Oja, Hopfield y las conclusiones para campañas siguen
pendientes. El MSE supervisado y los k-folds del TP3 no se trasladaron a este estudio.
