from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt

digits = datasets.load_digits()
# print(digits.data)
# print(digits.target)

clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data[:-1], digits.target[:-1])
print(clf.predict(digits.data[-3:]))

for i in range(1,4):
    plt.figure(i, figsize=(5, 5))
    plt.imshow(digits.images[-i], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()