# Data-analysis-challenge

This challenge aims at finding a classifier that can classify news articles into 24 classes with best possible accuracy. Please note
that this is a multi-class classification task, where each article has only one label.

# 1. Dataset description
The dataset in this competition is a large set of news articles that are crawled from a news website. The corpus contains 133,055 news articles in total. 80% of news articles are randomly selected for training. The other 20% are used for testing, the labels of which are withheld.

You can download this dataset from this Kaggle link: https://www.kaggle.com/patthoo/data-analysis-challenge-classifying-news-articles

# 2. Feature engineering and feature selection

## tfidf
After pre-processing the whole corpus, we then converted 21191 tokens to tf-idf, and removed 95% sparsity to get 329 features. However, tf-idf features only consider the importance of the word to a document in the whole set of corpora in terms of its frequency. Hence, we came to conclusion that using only tf-idf would loss the semantics of the word and the context of the whole document. That
is why we decided to utilise word embedding which will be discussed in the subsequent section.

## Word embedding
### Definition
Word embedding is technique to convert vocabulary in a large corpus into numeric vectors, particularly in this case, we will use global vectors for word representation (GloVe). GloVe is an unsupervised algorithm used to obtain vector representation of words by performing aggregated global word-word co-occurrence statistics from a corpus. This kind of representation shows some interesting properties such as linear substructures and the distance indicating relation between 2 nearest neighbour words. The Euclidean distance between 2 word vectors is an effective indicator to measure semantic similarity of those two corresponding words, for example distance between “man” - “sir” is larger than “man” - “woman”. Adding to the effective usage of GloVe is the linear substructure of word vectors, in which it is used to captured more intricate relationships of 2 words by introducing a vector difference between 2 vectors, because man and woman may be similar in terms of to describe human beings but intuitively they are semantically different. Thus, GloVe is designed to effectively capture as much as possible the meaning of a word by its juxtaposition to other words.

### Feature extraction
For word embedding, in pre-processing task, we only stopped at the step of removing all stopwords, numbers and punctuations. We do not proceed further steps such as stemming and lemmatization of tokens in a document because words (stored as a key) in Glove are maintained as their original form in document. For example, we have different vectors for “get”, “getting” and “gets” in Glove, so if all of them are converted to “get” during stemming and lemmatization, this can result in some information loss when we generate a vector for an entire document.

Then, we used a pre-trained word vectors file downloaded from GloVe to convert each document into a feature vector. The file consists of 840 billion vocabulary obtained by web crawling, and each of them is represented by a 300 dimensional vectors.

After extracting all word vectors from the file, we will take the sum of all of those vectors appearing in a document. Those documents which have empty text content or include meaningless text such as ‘20060316-closer’ or “20070808-closer-mid” are converted into a 300 dimensional vector of 0.

The features obtained after implementing above steps is a 300 dimensional vector representing each document. Then they are scaled before being fed into the model.

# 3. Methodology
The most important thing to watch out is the bias-variance trade off. If our model is too simple, it will likely to underfit the data and have high bias so the accuracy for both train and test will be low. On the other hand, as we increase training accuracy, our model gets more complicated and our model has high variance and will overfit the data, so the test accuracy will also be low. We need to find a point where the testing error is starting to increase and stop at this complexity. In order to see test accuracy, we use validation method cross validation.

We decided to use 5 fold cross validation to test our model accuracy. In 5 fold cross validation, the training set is divided into 5 folds of non-overlapping sections, and one fold is left out for validation of the model while the other 4 folds are used for training. The process is repeated 5 times so all 5 folds are used as validation set. The validation error is then averaged. Compared to validation set approach, 5 fold cv reduces the variability of the cv estimate and is less biased. Compared to leave one out cross validation, 5 fold cv is computationally cheaper to perform and has less variance. So in practice often 5 fold or 10 fold cv is used. We chose 5 for lower variance and cheaper computation.

Since this is a classification problem, there are a number of different models that can be used, such as logistic regression, knn, LDA, QDA, tree based models, svm, deep learning etc. Since we believe this is not a linear model, we decided to try knn, tree based models, svm and deep learning. The first model we tried was naive bayes, just to get a benchmark. We used naive bayes model with the tfidf features and with 5 fold cv and got 72% accuracy.

We also tried with other models such as SVM+tfidf, CNN, kNN + gloves. However, the accuracy rates only fall in the range from 70% to 73%.

The models with highest cross validation accuracy rate is SVM + GloVE. To improve the accuracy of our classifier, we also tuned in parameter and found the best parameters are 1 for gamma and 3 for cost.

The final accuracy rate for this model is 76%, outperforming other algorithms. Hence, SVM + GloVe is our final choice.

# 4. Model
SVM is shown to perform well for text classification. SVM is a generalisation of maximal margin classifier and is insensitive to the whole dataset, as it only cares about data points that are close to the decision boundaries.

In a maximal margin classifier, the margin is the distance between the decision boundary and the the closest data points to the decision boundary and the aim is to maximize this margin as much as possible so that the points are far away from the decision boundary. A maximal margin classifier is limited to linear separable data. The observations on the margins are known as support vectors.

Support Vector Classifier extends maximal margin classifier for the case of non separable datasets. We use a soft margin and define tuning parameter C, where C data points are allowed to cross the margin and the decision boundary. If C is large, the model is more biased and the variance is low, as C decreases the model becomes less biased and variance increases. We used tune.svm in R in order to find the best cost for cv.

Support Vector Machine introduces non linearity into decision boundaries and can be used for multiclass classification. This is done by having a non linear kernel. Several non linear kernels exist such as polynomial and radial. We chose to use radial kernel for more flexible decision boundaries and tuned the gamma value with tune.svm in R.

In summary, these theoretical evidence further support our choice of SVM:
a. SVM is very universal learners. As mentioned earlier, even though SVMs learn linear decision boundary in its basic form, it is still able to learn polynomial classifier, radial just by using an appropriate kernel function.
b. One of the properties of learning text classifier is that we have to deal with a large amount of features. The core element of SVM is the margin with which they separate the data, not the number of features. Its ability to learn regardless of the dimensionality of the input space suggests SVM should perform well for text classification.

# 5. References:
1. Joachims, T. (1998). Text categorization with Support Vector Machines: Learning with many relevant features. In Machine Learning: ECML-98, Tenth European Conference on Machine Learning, pp. 137-142, 1998, http://citeseer.nj.nec.com/joachims98text.html
2. Pennington J., Socher R., Manning C.D. (2014) Glove: Global Vectors for Word Representation, Empirical Methods in Natural Language Processing (EMNLP 2014)
