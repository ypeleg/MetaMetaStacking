
from mlxtend.regressor import StackingRegressor


class MetaMetaStacking(object):
    '''
    Stacked ensemble of stacked ensembles
    Only two layers is for pussies
    The arguments are lists of models, with the last argument being the first meta model. Then, each list of model
    is stacked with the next (backwards)
    '''
    def __init__(self, args):
        self._stack = self._build_stacked_ensemble(args)

    def _build_stacked_ensemble(self, metas):
        metas = list(metas)
        metas.reverse()
        current_meta = metas[0]
        for meta in metas[1:]:
            current_meta = StackingRegressor(meta, current_meta)
        return current_meta

    def fit(self, X, y):
        self._stack.fit(X, y)
        return self

    def predict(self, X):
        return self._stack.predict(X)
