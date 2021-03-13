from eye_detector.train.models import sgd, svc

MODELS = [
    svc.linear_svc,
    svc.svc,
    sgd.sgd,
    sgd.pca_sgd,
    sgd.rbg_sgd,
]

NAME_TO_MODEL = {
    model.__name__: model
    for model in MODELS
}
