from eye_detector.train.models import sgd, svc, torch, prob

MODELS = [
    svc.linear_svc,
    svc.svc,
    svc.rbf_svc,
    svc.pca_svc,
    sgd.sgd,
    sgd.pca_sgd,
    sgd.rbf_sgd,
    torch.torch,
    prob.gauss_rbf,
    prob.bayes,
]

NAME_TO_MODEL = {
    model.__name__: model
    for model in MODELS
}
