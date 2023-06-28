# Amazon Review Classifier Using PySpark

- **Project title:** Amazon Reviews Classifier Using Pyspark.
- **Dataset:** [Kaggle Amazon Reviews](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews?select=test.csv).
- **Technology:** `python` + `spark`.
- **Team:** CristÛbal Alc·zar, Yerko Garrido, Christopher Stears
- **Project Goal:** Create a term-frequency inverse document frequency (TF-IDF) feature matrix from the Amazon reviews dataset, and train a classification model to predict the product reviews.

## Abstract 

The increasing availability of large-scale datasets presents challenges in traditional machine learning approaches due to memory limitations and computational constraints. However, with Apache Spark's distributed computing capabilities, these challenges can be overcome. In this project, we aim to harness the power of PySpark, the Python library for Spark, to train a classifier on the [Kaggle Amazon Reviews](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews?select=test.csv) dataset, a massive collection of product reviews.

One of Spark's standout features is its ability to distribute computations across a cluster of machines, enabling the processing of datasets that exceed the memory capacity of a single machine. This distributed nature of Spark allows for parallel execution of tasks, making it an ideal platform for training machine learning models on extensive datasets. By utilizing PySpark's high-level API, MLlib, we can leverage its extensive set of distributed machine learning algorithms and tools.

Through this project, we aim to showcase the capabilities of PySpark and its MLlib library in tackling real-world machine learning challenges. By successfully training a sentiment classifier on the [Kaggle Amazon Reviews](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews?select=test.csv) dataset, we will demonstrate PySpark's ability to handle massive datasets, leverage distributed processing, and deliver scalable machine learning solutions. The insights gained from this project will not only advance our understanding of distributed machine learning but also open doors to scalable and efficient data analysis in various domains.


Here is an observation of the dataset:

> "2","Makes My Blood Run Red-White-And-Blue","I agree that every American should read this book -- and everybody else for that matter. I don't agree that it's scholarly. Rather, it's a joy to read -- easy to understand even for a person with two master's degrees! Between McElroy's chapter on How American Culture was Formed and Ken Burns' Lewis & Clark, I don't know which makes my blood run red-white-and-bluer. And as a child of the anti-establishment `60s, it's done a lot toward helping me understand why we Americans do what we do. It's the best history book I've ever read, the best history course I've ever taken or taught. I'm buying it for my home library for my grandchildren to use as a resource. We're also using it as a resource for a book on urban planning."

Each observation has three values: the polarity (1 for negative and 2 for positive), the heading and the body of the product review, respectively.


## Usage

For training the model, use the `train.py` script:

```bash
python3 train.py
```

Once the model is trained, you can load the pipeline with `inference.py` and 
get predictions. The output of this stage is saved in hdfs OUTPUT folder, and you can
check the file as follow:

```bash
hdfs dfs -cat /uhadoop2023/manco/amazon/predictions/part-00000-*.csv | head -n 10
hdfs dfs -cat /uhadoop2023/manco/amazon/predictions/part-00000-*.csv | wc
```

Then, you can copy the output from the hdfs to the local system:

```bash
hdfs dfs -get /uhadoop2023/manco/amazon/predictions/part-00000-*.csv part0.csv
```

## Data Preprocessing

The dataset is prepared by applying the following steps:

1. Combine the title and body columns into a single one.
2. Remove punctuation and any symbol different from letters and numbers.
3. Apply a tokenizer that separates the text by white spaces to build the vocabulary of the corpus.
4. Use the method CountVectorizer to count the number of each token per review.
5. Then, IDF performs the TF-IDF transformations based on the counts to compute the formula on the below image.
6. Finally, we get the feature matrix $N\times |V|$, in which $N$ is the number of observations, $|V|$ is the vocabulary size in the corpus, and each value in the matrix is the TF-IDF for a particular token in a product review.

<img src="./assets/tf-idf_card.png" width="450"/>

## Models

We trained two models for analyzing Amazon reviews: Logistic Regression and Classification Tree.

Logistic Regression:
Logistic regression is a statistical model commonly used for binary classification tasks. In the context of sentiment analysis for Amazon reviews, logistic regression is a suitable choice for predicting whether a review is positive or negative based on its features. By fitting a logistic function to the input data, logistic regression estimates the probability of a review belonging to a particular sentiment class. The model learns the relationship between independent variables (such as review text and its title) and the binary outcome (positive or negative sentiment). This allows us to quantify the impact of different features on the likelihood of a review being positive or negative.

Classification Tree:
A classification tree, also known as a decision tree, is a hierarchical structure that partitions the data into different categories based on feature values. In the context of analyzing Amazon reviews, a classification tree can be constructed to predict the sentiment or classify the reviews into distinct sentiment classes. The tree is built by recursively splitting the data using different features and their thresholds, creating a flowchart-like structure. Each internal node represents a decision based on a specific feature, guiding the review's path down the tree until it reaches a leaf node that corresponds to a sentiment class. Classification trees provide an interpretable approach to understanding the decision-making process and identifying the most influential features for sentiment classification.

## Results

Classification report for both models:

```
LogisticRegression:
              precision    recall  f1-score   support

           1       0.90      0.90      0.90    200328
           2       0.90      0.90      0.90    199672

    accuracy                           0.90    400000
   macro avg       0.90      0.90      0.90    400000
weighted avg       0.90      0.90      0.90    400000
```

```
ClassificationTree:
              precision    recall  f1-score   support

           1       0.63      0.79      0.70    157944
           2       0.84      0.69      0.76    242056

    accuracy                           0.73    400000
   macro avg       0.73      0.74      0.73    400000
weighted avg       0.75      0.73      0.73    400000
```

Confusion matrices for both models:

<table border=0 align=center>
    <tbody>
        <tr>
     	    <td width="50%" align="center"> <img src="./assets/confusion_matrix_lm.png"></td>
	    <td width="50%" align="center"> <img src="./assets/confusion_matrix_tree.png"></td>
        </tr>
    </tbody>
</table>


## Conclusions

1. PySpark seamlessly integrates Spark functionality with Python, enabling smooth interoperability with other machine learning frameworks and libraries.
1. Spark MLib offers a diverse range of distributed training models, including linear and tree-based models, eliminating the need to transfer intermediate steps between HDFS and the local system.
1. Despite the simplicity of the Amazon dataset (balance with 2 classes), impressive average F1 scores of 90% are achieved even without hyperparameter optimization (scales matter).

## Reference

```
@misc{CC5212-massive-manco,
  authors = {Alc√°zar, Cristbal}, {Garrido, Yerko}, {Stears, Christopher}
  title = {Project group 8: Amazon Review Classifier Using Pyspark},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/alcazar90/CC5212-massive}},
}
```
