from sklearn.preprocessing import FunctionTransformer


def to_dense_f(x):
    return x.toarray()


to_dense = FunctionTransformer(to_dense_f, accept_sparse=True)
