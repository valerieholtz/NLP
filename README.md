# NLP
### Collection of NLP related projects with Machine Learning and Deep Learning


- Different NLP models with TensorFlow/Keras on the IMDb Movie Reviews dataset
  - BoW
  - Fully Connected Neural Network
  - Word Embeddings
  - LSTM



- Sentiment Analysis on the yelp reviews dataset (https://www.kaggle.com/c/yelp-recsys-2013) with BoW: Feature Engineering and Linear Support Vector Classifier (SVC) give an accuracy of 0.84 with the following pipeline:

```
pipe_fe = Pipeline([('bow', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', LinearSVC())
                 ])
 ```



- Spam Filter with BoW: Multinomial Naive Bayes classifier and Random Forest Classifier on the UCI SMS Spam Collection Data Set (https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection). Data Preprocessing and training with the following pipeline:

```
pipeline_rf = Pipeline([
                     ('bow', CountVectorizer(analyzer=text_process)),
                     ('tfidf', TfidfTransformer()),
                     ('classifier', RandomForestClassifier())
                ])
```
