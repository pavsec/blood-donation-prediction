import load_data as ld
import util as ut

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def grid_search_mlp(model, X_train, Y_train):
    param_grid = {
        'activation': ['relu'],  # 'identity', 'logistic', 'tanh',
        'hidden_layer_sizes': [(400,)], # [(10, ), (100, ), (200, ), (300, ), (400, ), (500, ), (750, ), (1000, ), (1250, ), (1500, )],
        'solver': ['adam'],  # ['lbfgs', 'sgd', 'adam'],
        'alpha': [1e-5], #[1e-7, 1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-1, 3e-1],
        'learning_rate': ['adaptive'], #['constant', 'adaptive'],
        'max_iter': [1500]
    }

    ut.grid_search(model, X_train, Y_train, param_grid)

    return

if __name__ == '__main__':
    data = ld.load_data()
    X_train, X_test, Y_train, Y_test = ld.split_data(data, test_size=0.25)

    model = MLPClassifier(activation='relu', hidden_layer_sizes=(400, ), solver='adam', alpha=1e-5, learning_rate='adaptive', max_iter=1500)
    # grid_search_mlp(model, X_train, Y_train)
    
    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)

    cm = confusion_matrix(Y_test, predictions)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    print(classification_report(Y_test, predictions))

    print('Total MLP model accuracy: ' + str(accuracy_score(Y_test, predictions, normalize=True,  sample_weight=None)))
