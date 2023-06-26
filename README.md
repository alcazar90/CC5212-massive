# Amazon Review Classifier Using PySpark

## Abstract 

- **Project title:** Amazon Reviews Classifier Using Pyspark.
- **Dataset:** [Kaggle Amazon Reviews](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews?select=test.csv).
- **Technology:** `python` + `spark`.
- **Project Goal:** Create a term-frequency inverse document frequency (TF-IDF) feature matrix from the Amazon reviews dataset, and train a classification model to predict the product reviews.


Here is an observation of the dataset:

> "2","Makes My Blood Run Red-White-And-Blue","I agree that every American should read this book -- and everybody else for that matter. I don't agree that it's scholarly. Rather, it's a joy to read -- easy to understand even for a person with two master's degrees! Between McElroy's chapter on How American Culture was Formed and Ken Burns' Lewis & Clark, I don't know which makes my blood run red-white-and-bluer. And as a child of the anti-establishment `60s, it's done a lot toward helping me understand why we Americans do what we do. It's the best history book I've ever read, the best history course I've ever taken or taught. I'm buying it for my home library for my grandchildren to use as a resource. We're also using it as a resource for a book on urban planning."


Each observation has three values: the polarity (1 for negative and 2 for positive), the heading and the body of the product review, respectively.


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

Then, you can copy from the hdfs to the local system the output:

```bash
hdfs dfs -get /uhadoop2023/manco/amazon/predictions/part-00000-*.csv part0.csv
```

## Data Preprocessing

The dataset is prepared by applying the following steps:

1. Combine the title and body columns into a single one.
1. Remove punctuation and any symbol different from letters and numbers.
1. Apply a tokenizer that separates the text by white spaces to build the vocabulary of the corpus.
1. Use the method CountVectorizer to count the number of each token per review.
1. Then, IDF performs the TF-IDF transformations based on the counts to compute the following formula:
<center>
<img src="https://towardsdatascience.com/tf-term-frequency-idf-inverse-document-frequency-from-scratch-in-python-6c2b61b78558" alt="lala">
</center>
1. Finally, we get the feature matrix $N\times |V|$, in which $N$ is the number of observations, $|V|$ is the vocabulary size in the corpus, and each value in the matrix is the TF-IDF for a particular token in a product review.


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



## Reference

```
@misc{CC5212-massive-manco,
  authors = {Alcázar, Cristbal}, {Garrido, Yerko}, {Stears, Christopher}
  title = {Project group 8: Amazon Review Classifier Using Pyspark},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/alcazar90/CC5212-massive}},
}
```
