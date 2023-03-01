from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


def build_ai(data, labels):
    # train-test-split
    # random_state=0 --> no shuffling, because of random set
    # test_size=0.25 25% of the dataset is used for testing, the rest for training
    input_train, input_test, labels_train, labels_test = train_test_split(
        data, labels, random_state=1, test_size=0.25
    )

    # scale
    # optimizes the training set a bit (read up on)
    scaler = StandardScaler()
    scaler.fit(input_train)

    # DecisionTreeClassifier is best for classification tasks
    model = DecisionTreeClassifier()

    # model
    # model = LogisticRegression(max_iter=1000)

    # give the data to our model, and it already trains on it but there is nothing to see
    model.fit(input_train, labels_train)

    # print the accuracy of the model
    print(f"precision of model: {model.score(input_test, labels_test)}")
    tree.plot_tree(model)
    plt.show()
