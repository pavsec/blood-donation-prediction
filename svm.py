import load_data as ld
import util as ut

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay, cohen_kappa_score
import matplotlib.pyplot as plt


def grid_search_mlp(model, X_train, Y_train):
    param_grid = {
        'kernel': ['sigmoid'], #['rbf', 'poly', 'sigmoid', 'linear'] 'kernel': ['poly'],
        'gamma': ['auto'], #['auto', 'scale']
        'C': [0.3],#[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 10]
        # 'degree': [1, 2, 3, 4]
        'coef0': [1], #[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 10]
    }

    ut.grid_search(model, X_train, Y_train, param_grid)

    return


if __name__ == '__main__':
    data = ld.load_data()
    X_train, X_test, Y_train, Y_test = ld.split_data(data, test_size=0.25)

    model = SVC(kernel='sigmoid', gamma='auto', C=0.3)
    # grid_search_mlp(model, X_train, Y_train)

    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)

    cm = confusion_matrix(Y_test, predictions)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    print(classification_report(Y_test, predictions))

    print('Total SVC model accuracy: ' + str(accuracy_score(Y_test, predictions, normalize=True, sample_weight=None)))
    print('Total SVC model kappa score: ' + str(cohen_kappa_score(Y_test, predictions, sample_weight=None)))
