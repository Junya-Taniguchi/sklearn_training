# 高校入試の合格者の分類
# 教科は国語、数学、英語、社会、理科
# A高校合格には全て90点以上が必要
# B高校合格には全て70点以上が必要
# C高校合格には全て50点以上が必要
#
# A合格を1
# B合格を2
# C合格を3

from sklearn import svm
from sklearn.metrics import accuracy_score
import random

# 学習用のデータと結果を準備する
learn_data = [59, 41, 64, 84, 94],[57, 86, 68, 88, 56],[75, 77, 85, 99, 87],[70, 70, 91, 53, 40],[64, 94, 89, 44, 77],[84, 90, 74, 94, 74],[97, 99, 98, 90, 98],[60, 64, 64, 88, 49],[94, 97, 98, 96, 90],[79, 64, 76, 58, 66]
learn_label = [4,3,2,3,4,2,1,4,1,3]

# アルゴリズムを指定。SVM
clf = svm.SVC()  # 変更

# 学習用のデータと結果を学習する,fit()
clf.fit(learn_data, learn_label)

# テストデータによる予測,predict()
test_data = learn_data
test_label = clf.predict(test_data)

# テスト結果を評価する,accuracy_score()
print("予測対象：", test_data, ", 予測結果→", test_label)
print("正解率＝", accuracy_score(learn_label, test_label))

print(clf.predict(test_data))

# for i in range(10):
#     number = []
#     num  = random.uniform(40,100)
#     num1 = random.uniform(40, 100)
#     num2 = random.uniform(40, 100)
#     num3 = random.uniform(40, 100)
#     num4 = random.uniform(40, 100)
#     num = int(num)
#     num1 = int(num1)
#     num2 = int(num2)
#     num3 = int(num3)
#     num4 = int(num4)
#     number.extend([num,num1,num2,num3,num4])
#     print(number)
#
#
# # for i in range(100):
# #     print("テスト"+ str(i))
