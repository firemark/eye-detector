class BaseDecorator:

    def __init__(self, decoratee):
        self.__decoratee = decoratee

    def __getattr__(self, name):
        return getattr(self.__decoratee, name)

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)


class ModelDecorator(BaseDecorator):

    def eye_probability(self, vector):
        vector = vector.reshape(1, -1)
        score = self.predict(vector)[0]
        return max(1.0 - abs(score - 1.0), 0.0)


class ProbModelDecorator(BaseDecorator):

    def __init__(self, decoratee):
        self.decoratee = decoratee

    def __getattr__(self, name):
        return getattr(self.decoratee, name)

    def eye_probability(self, vector):
        vector = vector.reshape(1, -1)
        return self.predict_proba(vector)[0, 1]
