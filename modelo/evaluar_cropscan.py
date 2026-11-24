"""Evalúa un modelo CropScan (.tflite) contra la carpeta de prueba etiquetada por el equipo.

Estructura esperada de la carpeta de prueba:
    prueba/
        queresa/   (fotos con queresa confirmada por el ingeniero)
        sin_queresa/
Cada foto debe venir de una unidad productiva (cuartel) que NO se usó para entrenar.

Uso:
    python evaluar_cropscan.py cropscan.tflite prueba/ [umbral=0.5]
Salida: recall, precisión, F1 de queresa, F1 macro, matriz de confusión, intervalo de confianza 95 % del recall
y latencia por foto. Compara contra las metas del Charter: recall >= 0.85, F1 macro >= 0.85, <= 5 s.
"""
import sys, glob, os, time, math
import numpy as np, tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score
model, root = sys.argv[1], sys.argv[2]; thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
it = tf.lite.Interpreter(model_path=model); it.allocate_tensors()
i0, o0 = it.get_input_details()[0], it.get_output_details()[0]; H, W = i0['shape'][1:3]
def p(f):
    im = tf.image.resize(tf.io.decode_jpeg(tf.io.read_file(f), channels=3), (H, W)).numpy()[None].astype('float32')
    it.set_tensor(i0['index'], im); it.invoke(); return float(it.get_tensor(o0['index'])[0, 0])
X, y = [], []
for lab, d in ((1, 'queresa'), (0, 'sin_queresa')):
    for f in sorted(glob.glob(f'{root}/{d}/*.jp*g') + glob.glob(f'{root}/{d}/*.png')):
        X.append(f); y.append(lab)
y = np.array(y); t0 = time.time(); pr = np.array([p(f) for f in X]); lat = (time.time() - t0) / len(X)
yp = (pr >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
rec = tp / (tp + fn); prec = tp / (tp + fp) if tp + fp else 0.0
def wilson(k, n, z=1.96):
    ph = k / n; d = 1 + z * z / n; c = ph + z * z / (2 * n); r = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return (c - r) / d, (c + r) / d
lo, hi = wilson(tp, tp + fn)
f1 = f1_score(y, yp); f1m = f1_score(y, yp, average='macro')
print(f'Fotos: {len(y)} ({y.sum()} con queresa, {(1 - y).sum()} sin queresa), umbral {thr}')
print(f'Matriz [[VN, FP],[FN, VP]] = [[{tn}, {fp}],[{fn}, {tp}]]')
print(f'Recall queresa: {rec:.3f}  (IC 95 %: {lo:.3f} a {hi:.3f})  meta >= 0.85 -> {"CUMPLE" if lo >= 0.85 else ("cumple en punto, IC no concluyente" if rec >= 0.85 else "NO cumple")}')
print(f'Precisión: {prec:.3f}   F1 queresa: {f1:.3f}   F1 macro: {f1m:.3f}  meta >= 0.85 -> {"CUMPLE" if f1m >= 0.85 else "NO cumple"}')
print(f'Latencia: {lat * 1000:.0f} ms por foto en este equipo (medir también en el teléfono)  meta <= 5000 ms')
