import pandas as pd
import matplotlib.pyplot as plt

fruits = pd.read_table('fruits.txt')

#Understanding what the dataset looks like
print(fruits.head())
print(fruits.shape)
print(fruits["fruit_name"].unique())
print(fruits.groupby('fruit_name').size())

#A graphical rep of the dataset
import seaborn as sns
sns.countplot(fruits['fruit_name'], label='count')
plt.show()

#Box plot
fruits.drop('fruit_label', axis=1).plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False,
                                        figsize=(9,9), title='Box Plot for each input variable')
plt.savefig('Fruits_box')
plt.show()

#Plotting a Histogram
import pylab as pl
fruits.drop('fruit_label', axis=1).hist(bins=30, figsize=(9,9))
pl.suptitle('Histogram for each numeric input variable')
plt.savefig('Fruits_hist')
plt.show()

#Dividing the dataset into Target and Predictor Variables
features_names = ['mass', 'width', 'height', 'color_score']
x = fruits[features_names]
y = fruits['fruit_label']

#Creating the model with the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

#This helps to scale or normalize your data eg. mass is in 100+, height in 10+ and color score in decimals
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
print("Accuracy of Logistic Regression classifier on training set: {:.2f}".format(logreg.score(x_train, y_train)))
print("Accuracy of Logistic Regression classifier on testing set: {:.2f}".format(logreg.score(x_test, y_test)))

#DECISION TREE
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(x_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'.format(clf.score(x_train, y_train)))
print('Accuracy of Decision Tree classifier on testing set: {:.2f}'.format(clf.score(x_test, y_test)))

#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier().fit(x_train, y_train)
print('Accuracy of K Neighbor Classifier on training set: {:.2f}'.format(knn.score(x_train, y_train)))
print('Accuracy of K Neighbor classifier on testing set: {:.2f}'.format(knn.score(x_test, y_test)))

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(x_train, y_train)
print('Accuracy of GNB classifier on training set: {:.2f}'.format(gnb.score(x_train, y_train)))
print('Accuracy of GNB classifier on testing set: {:.2f}'.format(gnb.score(x_test, y_test)))

#Support Vector Machines
from sklearn.svm import SVC
svm = SVC().fit(x_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'.format(svm.score(x_train, y_train)))
print('Accuracy of SVM classifier on testing set: {:.2f}'.format(svm.score(x_test, y_test)))


gmma = "auto"
#CONFUSIION MATRIX for KNN Classifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = knn.predict(x_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))