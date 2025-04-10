from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


def compute_cross_val_score(x_train, y_train):
    print(
        "cross val score:",
        cross_val_score(GaussianNB(), x_train, y_train, cv=3),
    )


def print_info(y_test, y_pred):
    score = accuracy_score(y_test, y_pred)
    print("???" * 3)
    print("???", "score:", score)
    print("???", "confusion_matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("???")
    print("???", "classification_report:")
    print(classification_report(y_test, y_pred))
    print("???" * 3)
    return score
