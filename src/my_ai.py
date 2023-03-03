from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay


def build_ai(data, labels, print_conf_mat=False):
    input_train, input_test, labels_train, labels_test = train_test_split(
        data, labels, random_state=1, test_size=0.2
    )

    scaler = StandardScaler()
    scaler.fit(input_train)

    model = RandomForestClassifier()

    model.fit(input_train, labels_train)

    if print_conf_mat:
        disp = ConfusionMatrixDisplay.from_estimator(
            model,
            input_test,
            labels_test,
            normalize="true",
        )

        plt.show()

    # print the accuracy of the model
    return model.score(input_test, labels_test)
