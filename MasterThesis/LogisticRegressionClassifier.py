# python LogisticRegressionClassifier.py ngram+embedding TrainingSet.txt DevelopSet.txt Vectors2.txt char 4 >> result.txt

import sys
import sklearn.multiclass
import sklearn.linear_model
from   sklearn.feature_extraction.text import CountVectorizer
from   sklearn                         import preprocessing
import sklearn.metrics
import codecs
import numpy
import scipy.sparse

class Model:
    def __init__(self, 
                 vectorizer, 
                 classifier, 
                 embeddings={}, 
                 dimension=0,
                 features="ngram+embedding"):
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.embeddings = embeddings
        self.dimension  = dimension
        self.features = features

def tokenize(line):
    return line.split()

def parse(line):
    body, tags = line.split("\t")
    return (body, tags.split()) 

def read_embeddings(f):
    line = f.next()
    dim = int(line.split()[1])
    table = {}
    for line in f:
        fields = line.split()
        table[fields[0]] = numpy.array([ float(w) for w in fields[1:]])
    return table, dim

def main():
    features = sys.argv[1]
    trainfile = codecs.open(sys.argv[2], encoding='utf-8')
    testfile = codecs.open(sys.argv[3], encoding='utf8')
    vectorfile = codecs.open(sys.argv[4], encoding='utf-8')
    analyzetype = sys.argv[5]
    ngrammax = sys.argv[6]
    # Read training data and train model
    (bodies_train, tags_train) = zip(*(parse(line) for line in trainfile))
    table, dim = read_embeddings(vectorfile)
    # Read and preprocess test data
    (bodies_test, tags_test)   = zip(*(parse(line) for line in testfile))
    binarizer = make_binarizer(tags_train + tags_test)
    y_train = binarizer(tags_train)
    model = train(bodies_train, y_train, analyzetype, ngrammax, table, dim, features) 
    y_true = binarizer(tags_test)
    # Predict scores
    y_scores = predict_scores(model, bodies_test)
    # Print out Mean Average Precision
    print ("{0:.3f}".format(MAP(y_true, y_scores)))
    y_labels =  predict_labels(model, bodies_test)
    # Print out F1
    print ("{0:.3f}".format(sklearn.metrics.f1_score(y_true, y_labels, average='micro')))

def train(bodies, y_train, analyzetype, ngrammax, table, dim, features):
    vectorizer = CountVectorizer(input='content', 
                                 analyzer=analyzetype, 
                                 tokenizer=None,
                                 ngram_range=(1,int(float(ngrammax))),
                                 lowercase=False
                                 )
    X_train = combine(vectorizer.fit_transform(bodies), 
                        embeddings(bodies, table, dim),
                        features)
    print ('The result for', analyzetype,'type with ngram-range', ngrammax, 'and', sys.argv[1],'is:')
    classifier = sklearn.multiclass.OneVsRestClassifier(sklearn.linear_model.LogisticRegression())
    classifier.fit(X_train, y_train)
    return Model(vectorizer, classifier, table, dim, features)


def combine(X_1, X_2, features):
    if features == "ngram+embedding":
        print (X_2)
        gcv_mode="eigen"
        return numpy.concatenate((X_1.toarray(), X_2), axis=1)
    elif features == "ngram":
        return X_1
    elif features == "embedding":
        return X_2
    else:
        raise Exception("Wrong feature specification "+feature)

def embeddings(bodies, table, dim):
    def f(b):
        ws = tokenize(b)
        return sum( [ numpy.array(table.get(w, numpy.zeros(dim))) for w in ws ])
    return numpy.array([ f(b) for b in bodies ])

def make_binarizer(ys):
    classes = set()
    for y in ys:
        for y_i in y:
            classes.add(y_i)
    def binarizer(tagsets):
        result = numpy.array([[ (y in tagset) for y in classes  ] for tagset in tagsets ], 
                             dtype=int)
        return result
    return binarizer

def predict_scores(model, bodies):
    X_test = combine(model.vectorizer.transform(bodies), 
                     embeddings(bodies, model.embeddings, model.dimension),
                     model.features)
    y_scores = model.classifier.decision_function(X_test)
    return y_scores 

def predict_labels(model, bodies):
    X_test = combine(model.vectorizer.transform(bodies), 
                     embeddings(bodies, model.embeddings, model.dimension),
                     model.features)
    return model.classifier.predict(X_test)
    

def MAP(y_true, y_scores):
    pairs = zip(y_true, y_scores)
    results = [ sklearn.metrics.average_precision_score(y, y_score) for (y, y_score) 
                in pairs ]
    return mean(results)

def f1_score(y_true, y_pred):
    return sklearn.metrics.f1_score(y_true, y_pred, average='micro')

def mean(xs):
    return sum(xs)/len(xs)*1.0

if __name__ == '__main__':
    main()
    
