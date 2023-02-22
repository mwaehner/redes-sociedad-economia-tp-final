Trabajo final de la materia "Redes, Sociedad y Economía" dictada por Esteban Feuerstein

-------------------------------------

Para correr el proyecto:

Desde el directorio del proyecto, crear un nuevo virtual env:

`python3 -m venv .venv`

Activarlo:

`. .venv/bin/activate`

Instalar dependencias

`pip install -r requirements.txt`


Finalmente ejecutar con

`python run.py --df_dir="student-por.csv" --project_name portuguese --num_epochs=50 --seed=42 --dropout_rate=0.5 --hidden_size=30`

(datos de la clase de portugués)

o

`python run.py --df_dir="student-mat.csv" --project_name math --num_epochs=50 --seed=42  --dropout_rate=0.5 ----hidden_size=30`

(datos de la clase de la clase de matemática)

num_epochs, seed, dropout_rate y hidden_size se pueden variar como se desee.

-------------------------------------

Para visualizar los resultados a través de TensorBoard, desde el directorio del proyecto (portuguese-students/) hacer:


`python -m tensorboard.main --logdir=tb_logs`

Los modelos corresponden a las versiones:
version_0 = Hidden_10
version_1 = Hidden_30
version_2 = Hidden_30_droput

Se puede filtrar por "math" o "portuguese" para visualizar los resultados más fácilmente

-------------------------------------

Para replicar la experimentación del informe, correr:

`python experiment_script.py`

-------------------------------------

Matías Waehner
