import typing as th
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import string

DEFAULT_VECTOR_SIZE = 256
ENGLISH_NUMERICS = ''.join(str(i) for i in range(10))
ENGLISH_TRANSLATOR = str.maketrans('', '', ENGLISH_NUMERICS + string.punctuation)


class BasicPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.genre_labeler = LabelEncoder()
        self.popularity_labeler = LabelEncoder()

    def fit(self, x, y=None, **fit_params):
        self.genre_labeler.fit(x['genre'])
        self.popularity_labeler.fit(x['popularity'])
        return self

    def fit_transform(self, x, y=None, **fit_params):
        return self.fit(x, y, **fit_params).transform(x)

    def transform(self, x):
        result = x.copy()
        result['genre'] = self.genre_labeler.transform(x['genre'])
        result['popularity'] = self.popularity_labeler.transform(x['popularity'])
        result['info'] = (x['title'].map(str) + " " + x['plot'].map(str)).apply(
            lambda i: i.translate(ENGLISH_TRANSLATOR).lower())
        result['info-split'] = result['info'].apply(lambda i: i.split())
        result = result.drop(['rating', 'title', 'plot'], axis=1)
        return result


basic = BasicPreprocessor()

tf_idf_vectorizer = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('pca', TruncatedSVD())
])


class Vectorizer(BaseEstimator, TransformerMixin):
    def __init__(
            self, n_components: int = DEFAULT_VECTOR_SIZE, w2v: th.Optional[dict] = None,
            tfidf: th.Optional[dict] = None):
        """
        :param n_components: dimension size of vectors
        :param w2v: optional dictionary containing w2v-specific parameters
        :param tfidf: optional dictionary containing tfidf-specific pipeline parameters
        """
        # setting up parameters
        self.n_components = n_components
        self.w2v = w2v or dict(size=DEFAULT_VECTOR_SIZE, min_count=2)
        self.tfidf = tfidf or dict()

        # initializing vectorizers
        self.w2v_vectorizer = None
        self.tf_idf_vectorizer = tf_idf_vectorizer
        self.tf_idf_vectorizer.set_params(**self.tfidf)

    def __filter_w2vs(self, vecs):
        return [self.w2v_vectorizer.wv[j] for j in vecs if j in self.w2v_vectorizer.wv]

    def fit(self, x, y=None):
        # finalizing parameters
        self.w2v['size'] = self.w2v.get('size', self.n_components)
        self.tfidf['pca__n_components'] = self.tfidf.get(
            'pca__n_components', self.n_components)
        self.tf_idf_vectorizer.set_params(**self.tfidf)
        # fitting models
        self.tf_idf_vectorizer.fit(x['info'])
        self.w2v_vectorizer = Word2Vec(sentences=x['info-split'], **self.w2v)
        return self

    def transform(self, x):
        result = x.copy()
        result['vec_1'] = self.tf_idf_vectorizer.transform(x['info']).tolist()
        #         try:
        #             # the walrus operator is only available for python>=3.8
        #             # you can uncomment these lines if you are on python>=3.8
        #             result['vec_2'] = x['info-split'].apply(
        #                 lambda i: sum(z) / len(z) if (
        #                     z := [self.w2v_vectorizer.wv[j] for j in i if j in self.w2v_vectorizer.wv]) else 'NA'
        #             )
        #         except SyntaxError:
        result['vec_2'] = x['info-split'].apply(
            lambda i: sum(self.__filter_w2vs(i)) / len(self.__filter_w2vs(i)) if (self.__filter_w2vs(i)) else 'NA')
        result = result.drop(['info', 'info-split'], axis=1)
        return result

    def fit_transform(self, x, y=None, **fit_params):
        return self.fit(x, y).transform(x)


preprocessor = Pipeline(steps=[
    ('preprocess', BasicPreprocessor()),
    ('vectorizer', Vectorizer())
])
