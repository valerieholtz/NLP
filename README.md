# NLP
Collection of NLP related projects

- Spam Filter with Multinomial Naive Bayes classifier and Random Forest Classifier on the UCI SMS Spam Collection Data Set (https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection). Data Preprocessing and training with the following pipeline:

'''
pipeline_rf = Pipeline([
                     ('bow', CountVectorizer(analyzer=text_process)),
                     ('tfidf', TfidfTransformer()),
                     ('classifier', RandomForestClassifier())
                ])
                '''
