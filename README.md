# signaMed-server

Comandos despligue Signamed en los servidores:

#### SERVIDOR DE ACCESO : PEREIRO
```bash
PATH_ACCESS_SERVICE_FOLDER='/home/gts/projects/mvazquez/signamed-server/access_Service'
cd $PATH_ACCESS_SERVICE_FOLDER
nohup gunicorn --workers 5 --bind unix:http://unix:$PATH_ACCESS_SERVICE_FOLDER/app.sock app_access_version8:app > logs/gunicorn_app_access_verion8_$(date +"%Y_%m_%d_%H-%M").log &
```


#### SERVIDOR DE CÓMPUTO : BAIONA
```bash
PATH_COMPUTE_SERVICE_FOLDER='/home/temporal2/mvazquez/signamed-server/compute_Service'
cd $PATH_COMPUTE_SERVICE_FOLDER
nohup python app_compute.py > logs/app_compute_$(date +"%Y_%m_%d_%H-%M").log &
```

#### COMANDO / PIPELINE todo en uno - Procesado .json & inferir resultados.
```bash
cd utils/generate_prediction/src
python generate_prediction.py --input ../data/kps53/temp_afectar.json
```
Descripción del pipeline:
* Cargará el json situado en data/kps53 ( en donde he dejado varios .json de ejemplo)
* Parseará el .json extraerá los vectores de características (joints & bones)
* Despliega dos modelos de MSG3D (joints, bones) e infiere utilizando los vectores características del paso anterior.
* Combina la salida de ambos modelos y muestra resultados.

El resultado:
```bash
MS-G3D data_joints - prob: [0.8680568  0.03691714 0.03281879] & index: [ 2  4 13]
MS-G3D data_bones - prob: [0.99376476 0.00128119 0.00124165] & index: [2 4 1]
MS-G3D || Result :  prob: [[2, 4, 13]] & index: [[0.9737364649772644, 0.008597427047789097, 0.005205894820392132]]
```

