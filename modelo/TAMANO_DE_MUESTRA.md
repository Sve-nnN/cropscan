# Cuántas fotos de prueba hacen falta

Intervalo de confianza 95 % (Wilson) del recall según cuántas fotos con queresa tenga la carpeta de prueba.
Para afirmar "recall >= 85 %" con respaldo, el límite inferior del intervalo debe quedar sobre 0.85.

| Fotos con queresa | Recall observado | IC 95 % | ¿Límite inferior >= 0.85? |
|---|---|---|---|
| 20 | 0.90 | 0.70 a 0.97 | no |
| 20 | 0.95 | 0.76 a 0.99 | no |
| 30 | 0.90 | 0.74 a 0.97 | no |
| 30 | 0.93 | 0.79 a 0.98 | no |
| 50 | 0.90 | 0.79 a 0.96 | no |
| 50 | 0.96 | 0.87 a 0.99 | sí |
| 80 | 0.90 | 0.81 a 0.95 | no |
| 80 | 0.95 | 0.88 a 0.98 | sí |
| 100 | 0.90 | 0.83 a 0.94 | no |
| 100 | 0.95 | 0.89 a 0.98 | sí |
| 150 | 0.90 | 0.84 a 0.94 | no |
| 150 | 0.95 | 0.90 a 0.97 | sí |

Lectura: con 30 fotos positivas el límite inferior no pasa 0.85 aunque el recall observado sea 0.93. Con 50 o más fotos positivas y un recall observado de 0.95 o más, el límite inferior ya supera 0.85. Meta práctica: 50 a 80 fotos de queresa confirmada en la carpeta de prueba, de cuarteles distintos a los de entrenamiento.
