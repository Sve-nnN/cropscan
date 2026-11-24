"""CropScan: prototipo de clasificador (MobileNetV2) -> TFLite, con evaluación completa.
Uso: python train.py <carpeta_raw> <salida>
Estructura de <carpeta_raw>: una subcarpeta por clase original; CLASES_POS define cuáles cuentan como positivo.
"""
import sys, os, json, glob, random, time
import numpy as np, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, classification_report
SEED=42; random.seed(SEED); np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)
RAW, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
CLASES_POS={'Scale_insect','Mealy_bugs'}
IMG=224
files,labels,orig=[],[],[]
for d in sorted(glob.glob(f'{RAW}/*')):
    c=os.path.basename(d)
    for f in sorted(glob.glob(d+'/*.jpg')):
        files.append(f); labels.append(1 if c in CLASES_POS else 0); orig.append(c)
files=np.array(files); labels=np.array(labels); orig=np.array(orig)
print('total',len(files),'pos',labels.sum(),'neg',(1-labels).sum())
strat=np.array([f'{o}' for o in orig])
tr,te,ytr,yte,otr,ote=train_test_split(files,labels,orig,test_size=0.2,random_state=SEED,stratify=strat)
tr,va,ytr,yva,otr,ova=train_test_split(tr,ytr,otr,test_size=0.15,random_state=SEED,stratify=otr)
print('train',len(tr),'val',len(va),'test',len(te),'test pos',yte.sum())
def load(f,y,aug=False):
    im=tf.io.decode_jpeg(tf.io.read_file(f),channels=3); im=tf.image.resize(im,(IMG,IMG))
    if aug:
        im=tf.image.random_flip_left_right(im); im=tf.image.random_flip_up_down(im)
        im=tf.image.random_brightness(im,0.15); im=tf.image.random_contrast(im,0.85,1.15)
    return im,y
def ds(f,y,aug=False,bs=32):
    d=tf.data.Dataset.from_tensor_slices((f,y.astype('float32')))
    if aug: d=d.shuffle(len(f),seed=SEED)
    return d.map(lambda a,b:load(a,b,aug),num_parallel_calls=4).batch(bs).prefetch(2)
base=tf.keras.applications.MobileNetV2(input_shape=(IMG,IMG,3),include_top=False,weights='imagenet',alpha=1.0)
base.trainable=False
inp=tf.keras.Input((IMG,IMG,3),name='foto_0_255')
x=tf.keras.applications.mobilenet_v2.preprocess_input(inp)
x=base(x,training=False); x=tf.keras.layers.GlobalAveragePooling2D()(x); x=tf.keras.layers.Dropout(0.3)(x)
out=tf.keras.layers.Dense(1,activation='sigmoid',name='p_queresa')(x)
m=tf.keras.Model(inp,out)
neg,pos=(1-ytr).sum(),ytr.sum(); cw={0:len(ytr)/(2*neg),1:len(ytr)/(2*pos)}
m.compile(tf.keras.optimizers.Adam(1e-3),'binary_crossentropy',metrics=['accuracy',tf.keras.metrics.Recall(name='recall')])
cb=[tf.keras.callbacks.EarlyStopping(patience=4,restore_best_weights=True,monitor='val_loss')]
m.fit(ds(tr,ytr,True),validation_data=ds(va,yva),epochs=12,class_weight=cw,callbacks=cb,verbose=2)
base.trainable=True
for l in base.layers[:-30]: l.trainable=False
m.compile(tf.keras.optimizers.Adam(1e-5),'binary_crossentropy',metrics=['accuracy',tf.keras.metrics.Recall(name='recall')])
m.fit(ds(tr,ytr,True),validation_data=ds(va,yva),epochs=8,class_weight=cw,callbacks=cb,verbose=2)
# umbral elegido en validación para recall >= 0.90 (prioriza no perder positivos)
pv=m.predict(ds(va,yva),verbose=0).ravel()
thr=0.5
for t in np.arange(0.9,0.05,-0.025):
    if recall_score(yva,(pv>=t).astype(int))>=0.90: thr=float(t); break
print('umbral',thr)
# evaluación en test con el modelo Keras
pt=m.predict(ds(te,yte),verbose=0).ravel(); yp=(pt>=thr).astype(int)
# exportar TFLite (cuantización dinámica) y evaluar el .tflite
conv=tf.lite.TFLiteConverter.from_keras_model(m); conv.optimizations=[tf.lite.Optimize.DEFAULT]
tfl=conv.convert(); open(f'{OUT}/cropscan_proto.tflite','wb').write(tfl)
it=tf.lite.Interpreter(model_content=tfl); it.allocate_tensors()
i0=it.get_input_details()[0]; o0=it.get_output_details()[0]
def run(f):
    im=tf.image.resize(tf.io.decode_jpeg(tf.io.read_file(f),channels=3),(IMG,IMG)).numpy()[None].astype('float32')
    it.set_tensor(i0['index'],im); it.invoke(); return float(it.get_tensor(o0['index'])[0,0])
t0=time.time(); pl=np.array([run(f) for f in te]); lat=(time.time()-t0)/len(te)*1000
ypl=(pl>=thr).astype(int)
def rep(y,yp,name):
    cm=confusion_matrix(y,yp,labels=[0,1])
    return {'nombre':name,'recall_positivo':float(recall_score(y,yp)),'precision_positivo':float(precision_score(y,yp,zero_division=0)),
     'f1_positivo':float(f1_score(y,yp)),'f1_macro':float(f1_score(y,yp,average='macro')),'matriz_[[TN,FP],[FN,TP]]':cm.tolist()}
res={'n_total':int(len(files)),'n_train':int(len(tr)),'n_val':int(len(va)),'n_test':int(len(te)),'n_test_pos':int(yte.sum()),
     'umbral':thr,'keras':rep(yte,yp,'keras'),'tflite':rep(yte,ypl,'tflite'),
     'tflite_tamano_MB':round(len(tfl)/1e6,2),'latencia_ms_por_foto_CPU_Mac_M2':round(lat,1),
     'por_clase_original_test':{c:{'n':int((ote==c).sum()),'aciertos':int((yp[ote==c]==yte[ote==c]).sum())} for c in sorted(set(ote))}}
json.dump(res,open(f'{OUT}/metricas.json','w'),indent=2,ensure_ascii=False); print(json.dumps(res,indent=2,ensure_ascii=False))
