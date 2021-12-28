# machine-learning-and-statistics
Machine Learning and Statistics @ GMIT 2021

Assessment (100%)
GitHub Repository (20%):
The repository should contain the following:
10%: A clear and informative README.md, explaining why the repository exists, what’s in it, and how to run the notebooks.
10%: A requirements.txt file that enables someone to quickly run your notebooks with minimal configuration. You should also include any other required files such as data files and image files.

Scikit-Learn Jupyter Notebook (40%):
Include a Jupyter notebook called scikit-learn.ipynb that contains the following.
10%: A clear and concise overview of the scikit-learn Python library.
20%: Demonstrations of three interesting scikit-learn algorithms.
10%: Appropriate plots and other visualisations to enhance your notebook for viewers.

Overview of Scikit-Learn:
Scikit-Learn is a library in Python that provides users with a set of tools for machine learning and statistical modelling purposes.
Many tools that can be utilised with the Python interface Scikit-Learn include classification, regression and clustering/biclustering.
Scikit-Learn is primarily written in Python and is also built upon other libraries in Python including numpy, scipy and matplotlib, with numerous supervised and unsupervised learning algorithms that can be implemented by data analysts.
For this assignment, the following three Scikit-Learn algorithms will be analysed and discussed:
    #1: Decision Trees
    #2: Clustering/Biclustering
    #3: Manifold Learning

#1: Decision Trees:
Decision Tree (DT) is a supervised machine learning method used in classification and regression. DTs work by creating a model that's capable of predicting the value of a target variable from learning simple decision rules inferred from the data features.
DTs learn from the data to approximate a sine curve with a set of simple decision rules, if-then-else, where the deeper the tree, the more complex the decision rules.

Advantages of Decision Trees:
    DTs are simple and straightforward to understand and interpret, as DTs can be visualised.

    DTs requires little data preparation, where other techniques often require data normalisation, dummy variables would need to be created and blank values also need to be removed.

    The cost of using DTs (predicting the data) is logarithmic in the number of data points used to train the DT.

    DTs are capable of handling multi-output problems.

    DTs use a white box model where if a given situation is observable, Boolean logic explains the observed condition. In contrast to a black box model, results may be more difficult to interpret.

Disadvantages of Decision Trees:
    DT learners can create overly complex trees that do not generalise the dataset very well, known as overfitting. Overfitting can only be avoided by the use of mechanisms such as DT pruning, setting the maximum depth of the tree and setting the minimum number of samples required at a leaf node.

    DTs can be unstable due to small variations in the dataset, which can potentially result in a completely unexpected DT being generated.

    The predictions of DTs are seen as neither smooth nor continuous, so these are not suitable for data extrapolation.

    There are concepts such as parity in DTs which DTs don't express easily.

Decision Tree Implementation:
DecisionTreeClassifier() is an imported class in Python that's capable of performing multi-class classification on a dataset.
DecisionTreeClassifier() takes in two arrays as input:
    X, sparse or dense, of shape (n_samples, n_features)
    holding the training samples.
    Y, of integer values, of shape (n_samples,), holding the
    class labels for the training samples.
After the class has been fitted, the model can then predict the class of samples.
In the case of multiple classes that have the same and the highest probability, the DecisionTreeClassifier() predicts the class with the lowest index amongst the multiple classes.
As an alternative to outputting this specific class, the probability of each class can be predicted, a fraction of training samples of the class within a leaf on the Decision Tree.
DecisionTreeClassifier() is capable of both binary classification (where labels are [-1, 1]) and multiclass classification (where labels are [0, …, K-1]).
Using the Iris dataset, we can construct a Decision Tree.
Once trained with the Iris dataset, a Decision Tree can be plotted via the plot_tree() function.
The Decision Tree can be exported via Graphviz exporter, export_graphviz, and saved into an external pdf file.
The export_graphviz exporter is capable of a variety of aesthetic options, including colouring nodes by their class or value for regression, using explicit variable and class names. The plots are automatically modified within Jupyter notebooks.
The Decision Tree can also be exported in textual format via the export_text() function.

Scipy-Stats Jupyter Notebook (40%):
Include a Jupyter notebook called scipy-stats.ipynb that contains the following:
10%: A clear and concise overview of the scipy.stats Python library.
20%: An example hypothesis test using ANOVA. You should find a data set on which it is appropriate to use ANOVA, ensure the assumptions underlying ANOVA are met, and then perform and display the results of your ANOVA using scipy.stats.
10%: Appropriate plots and other visualisations to enhance your notebook for viewers.

References:
https://machinelearningmastery.com/a-gentle-introduction-to-scikit-learn-a-python-machine-learning-library/
https://www.tutorialspoint.com/scikit_learn/index.htm
https://www.tutorialspoint.com/scikit_learn/scikit_learn_decision_trees.htm
https://www.analyticsvidhya.com/blog/2015/01/scikit-learn-python-machine-learning-tool/
https://medium.com/analytics-vidhya/most-used-scikit-learn-algorithms-part-1-snehit-vaddi-7ec0c98e4edd
https://scikit-learn.org/stable/supervised_learning.html
https://scikit-learn.org/stable/modules/tree.html
https://scikit-learn.org/stable/unsupervised_learning.html
https://scikit-learn.org/stable/modules/clustering.html
https://scikit-learn.org/stable/modules/biclustering.html