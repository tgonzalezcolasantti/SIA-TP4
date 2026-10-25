# Hopfield: memoria asociativa

Implementación independiente de Kohonen. No cambia los modelos existentes ni sus
experimentos. Usa las dependencias del proyecto, sin agregar un lector externo de QR.

## Ejecución

Desde la raíz del TP4:

```powershell
uv run python -m experiments.run_hopfield
```

Para ejecutar las partes por separado:

```powershell
uv run python -m experiments.run_hopfield --mode letters
uv run python -m experiments.run_hopfield --mode qr
```

Se admite `--config configs/hopfield.json` y `--no-plots` para exportar solamente
datos, historiales y métricas. Las rutas del JSON son relativas al propio JSON.
Los resultados se guardan en `results/hopfield`, que Git ignora.

## Modelo y representación

- Cada letra es un patrón de 25 neuronas, en una grilla de 5 x 5. Se almacenan
  cuatro letras configurables. El conjunto ampliado para estudiar capacidad tiene
  ocho letras, definidas en `experiments/hopfield_data.py`.
- Cada QR se lee tal como está en el repositorio, con 37 x 37 píxeles. No se
  redimensiona ni interpola. Negro se representa con +1 y blanco con -1.
- Se almacenan simultáneamente los cinco QR en una red de 1369 neuronas.
- Hebb: `W = patterns.T @ patterns / N`, con diagonal cero y pesos simétricos.
- La recuperación es asincrónica: visita todas las neuronas en un orden aleatorio
  nuevo por barrido. Ante campo nulo se conserva el valor anterior, con tolerancia
  numérica de 1e-12. No se implementa actualización sincrónica.
- Un barrido completo sin cambios confirma convergencia. Llegar al máximo de
  barridos se informa como `not_converged`, aunque el último estado pudiera estar
  cerca de una memoria. No se confunde ese caso con un estado espurio.
- La energía es `-0.5 * s.T @ W @ s`. Se calcula inicialmente y se actualiza usando
  `delta_E = -delta_s_i * h_i` en cada cambio de neurona. El campo local se actualiza
  incrementalmente para evitar un producto matricial completo por cada neurona.

## Experimentos

La configuración predeterminada prueba 0, 5, 10, 20, 30, 40 y 50% de ruido con tres
semillas para cada patrón. Se invierten exactamente `floor(fracción * N)` posiciones
distintas. Es la adaptación bipolar del ruido del TP3: `x -> -x`, en lugar de `1-x`.

También se simula una esquina superior izquierda borrada a blanco. La fracción
configurada describe el **lado** del cuadrado borrado, no la cantidad de bits
invertidos. `hamming_input` registra cuántos bits cambiaron realmente. No se modela
una entrada con valores desconocidos: los píxeles borrados pasan a ser blancos.

Se clasifican los resultados como:

| Estado | Significado |
| --- | --- |
| `correct` | Convergió al patrón original consultado |
| `other_memory` | Convergió a otra memoria almacenada |
| `inverse_memory` | Convergió al negativo de una memoria almacenada |
| `spurious` | Convergió a un estado que no coincide con ninguna memoria ni su negativo |
| `not_converged` | Alcanzó el límite sin confirmar estabilidad |

Los negativos se reportan por separado aunque también son recuerdos no deseados.
El estudio de letras busca explícitamente un estado espurio partiendo de patrones
con mucho ruido. Si no lo encuentra dentro del límite configurado, lo informa;
no inventa un ejemplo ni etiqueta automáticamente cualquier fallo como espurio.

El estudio de capacidad almacena prefijos de 1 a 8 letras, consulta **todas** las
letras almacenadas con el mismo nivel de ruido y agrega las semillas. Es un
experimento empírico para ese orden de incorporación, no una demostración de una
capacidad universal. El solapamiento normalizado `xi @ xj / N` permite analizar
ortogonalidad: cero implica ortogonalidad y magnitudes altas indican similitud.
Con 25 entradas bipolares, un producto interno entre dos letras no puede ser cero
exactamente porque es la suma de una cantidad impar de términos +1/-1.

## Salidas

En cada subcarpeta `letters/` y `qr/`:

| Archivo | Contenido |
| --- | --- |
| `stored_patterns.npz`, `stored_patterns.png` | Patrones originales y etiquetas |
| `overlap.csv`, `overlap.png` | Matriz de solapamiento para analizar ortogonalidad |
| `stored_stability.csv` | Si cada memoria original es un punto fijo y su energía |
| `all_trials.csv` | Resultado, semillas, Hamming, convergencia, barridos y energía de cada consulta |
| `recall_traces.csv` | Energía y cantidad de cambios por barrido en todas las consultas |
| `queries_and_results.npz` | Consultas y estados finales; filas en el orden de `trial_id` |
| `summary.csv`, `per_pattern_summary.csv` | Tasas de éxito, convergencia y métricas agregadas |
| `outcomes.csv` | Conteos por tipo de resultado |
| `summary.png` | Recuperación, barridos y Hamming según deterioro |
| `examples/` | Ejemplos con imágenes, historia de estados y energía por actualización individual |

Los ejemplos son las consultas de la primera semilla para el primer patrón a
todos los niveles y para todos los patrones al nivel central de ruido. Se eligen
por configuración, sin seleccionar solamente los casos exitosos.

`letters/` incluye además `capacity_trials.csv`, `capacity_summary.csv`,
`capacity.png`, `spurious_search.csv` y `spurious_search.json`. Si encuentra un
estado espurio, exporta su consulta, estado final, historial e ilustración.

En la carpeta principal se guardan configuración, versiones, convenciones del
experimento y hashes de los PNG originales. Para comparar estudios, usar distintas
carpetas `out_dir`; repetir en la misma carpeta sobrescribe archivos y puede dejar
ejemplos viejos, que se deben contrastar con los `trial_id` del estudio actual.

## Limitaciones e interpretación

Una energía menor no garantiza recuperar el QR correcto: un mínimo puede ser otra
memoria o un estado espurio. Los QR comparten estructura, por lo que no cumplen
necesariamente el supuesto de patrones aleatorios poco correlacionados. Tener
solamente cinco patrones para 1369 neuronas no garantiza recuperarlos todos.
Las reglas de capacidad aproximada de Hopfield dependen de supuestos estadísticos;
con letras y QR se debe mirar la estabilidad y recuperación observadas.

La tasa de éxito exige coincidencia exacta con el patrón original y convergencia.
**No se ejecuta un decodificador de QR**: los PNG recuperados de 37 x 37 se exportan
para probarlos con un lector. La distancia de Hamming no equivale a legibilidad,
porque un lector puede corregir ciertos errores y fallar ante otros. Las pruebas
de ruido independiente y esquina blanca no abarcan rayones, perspectiva, manchas
o errores de captura de una foto real. La red solo recuerda los menús almacenados.

## Resultado de la configuración inicial

La ejecución con las semillas 42, 43 y 44 y la configuración incluida produjo:

| Ruido aleatorio | Recuperación exacta de letras | Recuperación exacta de QR |
| --- | --- | --- |
| 0% | 12/12 | 0/15 |
| 5% | 12/12 | 0/15 |
| 10% | 12/12 | 0/15 |
| 20% | 12/12 | 0/15 |
| 30% | 10/12 | 0/15 |
| 40% | 3/12 | 0/15 |
| 50% | 1/12 | 0/15 |

Los cinco QR originales son inestables según `stored_stability.csv`: la red los
modifica incluso sin deterioro. En los ensayos de 0 a 40% de ruido, la distancia
media final es de 143 bits. Por lo tanto, la disminución de energía y la convergencia
no implican restaurar el menú correcto. El resultado es compatible con interferencia
entre patrones que comparten estructura; el solapamiento se exporta para analizarla.
No se cambió la regla de aprendizaje ni se seleccionaron QR más fáciles para evitar
este resultado.

Con las letras se encontró un estado espurio estable en el noveno intento de la
búsqueda (índice 8), a partir de A con 12 de sus 25 bits invertidos. El estado final
difiere de las cuatro memorias y de sus negativos. La recuperación empírica con
20% de ruido cae de 3/3 ensayos con una letra almacenada a 5/24 con ocho letras;
no es estrictamente monótona para cada incorporación y no establece un límite
teórico universal.

Estas cifras describen la configuración incluida y pocas semillas; no son una
estimación exhaustiva. Queda integrar los gráficos y discutir las conclusiones en
la entrega del TP, especialmente la diferencia entre recuperación exacta y lectura
efectiva por un decodificador de QR.

## Verificación

```powershell
uv run python -m unittest discover -s tests -v
```

Los tests verifican Hebb, simetría, diagonal cero, recuperación, energía en cada
actualización, empates, corte sin convergencia, ruido exacto, representación de QR,
ausencia de overflow en solapamientos y reproducción de un estudio con ambos casos.
