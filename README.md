# Amazon Review Classifier Using PySpark

- Project title: Amazon reviews sentiment classifier.
- Data used: Kaggle Amazon Reviews.
- Technology used: Spark.
- Main objective: Create a term-frequency inverse document frequency (TF-IDF) feature matrix from amazon reviews and train a binary sentiment classifier (i.e. positive / negative).


A random amazon review:

> "2","Makes My Blood Run Red-White-And-Blue","I agree that every American should read this book -- and everybody else for that matter. I don't agree that it's scholarly. Rather, it's a joy to read -- easy to understand even for a person with two master's degrees! Between McElroy's chapter on How American Culture was Formed and Ken Burns' Lewis & Clark, I don't know which makes my blood run red-white-and-bluer. And as a child of the anti-establishment `60s, it's done a lot toward helping me understand why we Americans do what we do. It's the best history book I've ever read, the best history course I've ever taken or taught. I'm buying it for my home library for my grandchildren to use as a resource. We're also using it as a resource for a book on urban planning."


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
  title = {Project group 8: Amazon Review Classifier Using Pyspark},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/alcazar90/CC5212-massive}},
}
```
