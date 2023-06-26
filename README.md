# Amazon Review Classifier Using PySpark

## Usage

For training the model use the `train.py` script:

```bash
python3 train.py
```

Once the model is trained, you can load the pipeline with `inference.py` an 
get predictions. The output of this stage is saved in:

```bash
hdfs dfs -cat /uhadoop2023/manco/amazon/predictions/part-00000-*.csv | wc
```

## TODO

1. [X] Cargar los datos.
1. [X] Cambiar los nombres (?), los archivos no vienen con nombre creo...
1. [X] Concatenar `review_body` y `review_title` en una sola columna.
1. [X] Tokenizar, computar TF-IDF, luego entrenar una regresino logistica
1. [X] Guardar pipeline de prediccion
1. [X] Crear `inference.py` para utilizar el pipeline de predicciion
1. [ ] Escalar entrenamiento
1. [ ] Archivo de analisis y computar metrica
1. [ ] Entrenar otra clase de modelo diferente tipo arboles...



## Dataset

[Kaggle Amazon review dataset](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews?select=test.csv)


```
@misc{CC5212-massive-manco,
  authors = {Alcázar, Cristól}, {Garrido, Yerko}, {Stears, Christopher}
  title = {Amazon Review Classifier Using Pyspark},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/alcazar90/CC5212-massive}},
}
```
