# CropScan

App Android que detecta queresa en palta con una CNN que corre en el celular (TFLite). Informa, no diagnostica.
Proyecto del curso Gerencia de Proyectos de Computación, UPC, Grupo 04.

## Contenido

- `modelo/entrenar_prototipo.py`: entrena MobileNetV2 (transferencia desde ImageNet) y exporta TFLite.
- `modelo/evaluar_cropscan.py`: mide recall con intervalo de confianza, precisión, F1, matriz de confusión y latencia.
- `modelo/TAMANO_DE_MUESTRA.md`: cuántas fotos de prueba hacen falta para respaldar un recall de 85 %.
- `modelo/LEEME.md` y `modelo/metricas.json`: estado y límites del prototipo.

## Estado

Prototipo entrenado con plagas de agarwood (otro cultivo). Sirve para probar el flujo, el tamaño y la velocidad.
Sus métricas no son la meta del Charter. El recall real requiere fotos de queresa en palta confirmadas por el ingeniero.
