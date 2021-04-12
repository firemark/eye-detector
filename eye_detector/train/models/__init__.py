from eye_detector.train.models import sgd, svc, torch, prob, ridge

MODELS = [
    svc.linear_svc,
    svc.svc,
    svc.rbf_svc,
    svc.pca_svc,
    sgd.sgd,
    sgd.sgd_log,
    sgd.sgd_elastic,
    sgd.pca_sgd,
    sgd.rbf_sgd,
    torch.torch,
    prob.gauss_rbf,
    prob.bayes,
    ridge.ridge,
]

NAME_TO_MODEL = {
    model.__name__: model
    for model in MODELS
}
