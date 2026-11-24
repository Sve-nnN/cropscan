# Modelo CropScan: prototipo y kit de evaluación

## Qué es y qué NO es
Prototipo para probar el pipeline completo (entrenar, exportar a TFLite, medir tamaño y velocidad, evaluar con métricas). **No es el modelo de queresa en palta y sus métricas NO cuentan para la meta del Charter** (recall de queresa >= 85 % y F1 macro >= 0.85 en prueba independiente con fotos propias).

## Con qué se entrenó
Dataset público de Mendeley "Agarwood Leaf Image Dataset for Pest and Disease Analysis" (hojas de agarwood, otro cultivo). Muestra de 487 fotos: positivo = "insecto tipo escama o cochinilla" (37 Scale insect + 150 Mealy bugs, que son cochinilla harinosa, no queresa), negativo = otras plagas (100 Spider & webs, 100 Flea Beetles, 100 Brown clumps). Sin clase de hoja sana. Licencia: revisar la ficha del dataset antes de citarlo.

Arquitectura: MobileNetV2 (ImageNet) con transferencia, 224x224, ajuste fino de las últimas 30 capas, cuantización dinámica a TFLite. Partición aleatoria 330 / 59 / 98 (entrenamiento / validación / prueba), umbral 0.675 elegido en validación.

## Resultado del prototipo (98 fotos de prueba, 38 positivas)
| Medida | Keras | TFLite |
|---|---|---|
| Recall del positivo | 0.816 | 0.789 |
| Precisión | 0.939 | 0.938 |
| F1 del positivo | 0.873 | 0.857 |
| F1 macro | 0.901 | 0.889 |
| Matriz [[VN, FP],[FN, VP]] | [[58, 2],[7, 31]] | [[58, 2],[8, 30]] |

- Tamaño del `.tflite`: 2.53 MB.
- Latencia: 6.3 ms por foto en CPU de un Mac M2. Falta medir en un teléfono Android real; la meta es 5 s o menos.
- El recall del prototipo (0.79) queda bajo 0.85 incluso en estas condiciones favorables. Con 38 positivas, su intervalo de confianza es amplio.

## Límites que hay que decir en voz alta
1. Otro cultivo y otra plaga: no dice nada sobre queresa en palta.
2. Partición aleatoria, no por unidad productiva: puede haber fotos casi iguales entre entrenamiento y prueba, lo que infla las métricas. El Charter exige separar por unidad productiva.
3. Sin hojas ni frutos sanos como negativo.
4. Solo 37 fotos de escama verdadera.

## Qué falta para tener el modelo real
1. Fotos de queresa en palta, confirmadas por el ingeniero, de varios cuarteles (meta del Charter: 600 o más en total, con negativos sanos).
2. Entrenar con `entrenar_prototipo.py` ajustando `RAW` y `CLASES_POS` y partiendo por cuartel.
3. Evaluar con `evaluar_cropscan.py` sobre una carpeta de prueba de cuarteles que no se usaron para entrenar (50 a 80 fotos con queresa, ver `TAMANO_DE_MUESTRA.md`).
4. Medir la latencia en el teléfono.
