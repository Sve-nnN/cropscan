import random, io, os
from remotezip import RemoteZip
from PIL import Image
random.seed(7)
u=open('url.txt').read().strip()
plan={'Scale insect':37,'Mealy bugs':150,'Spider & webs':100,'Flea Beetles':100,'Brown clumps':100}
os.makedirs('raw',exist_ok=True)
with RemoteZip(u, headers={'User-Agent':'Mozilla/5.0'}) as z:
    names=[n for n in z.namelist() if not n.endswith('/')]
    for cls,k in plan.items():
        fs=[n for n in names if n.split('/')[-2]==cls]
        fs=random.sample(fs,min(k,len(fs)))
        d=f"raw/{cls.replace(' ','_').replace('&','y')}"; os.makedirs(d,exist_ok=True)
        for i,n in enumerate(fs):
            try:
                im=Image.open(io.BytesIO(z.read(n))).convert('RGB'); im.thumbnail((448,448)); im.save(f"{d}/{i:03d}.jpg",quality=90)
            except Exception as e: print('skip',n,e)
        print(cls,len(fs),flush=True)
print('DONE')
