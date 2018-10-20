from sklearn import svm  # 変更
from sklearn.metrics import accuracy_score

# 学習用のデータと結果を準備する
learn_data = [[0, 0], [1, 0], [0, 1], [1, 1]]
learn_label = [0, 1, 1, 0]

# アルゴリズムを指定。SVM
clf = svm.SVC()  # 変更

# 学習用のデータと結果を学習する,fit()
clf.fit(learn_data, learn_label)

# テストデータによる予測,predict()
test_data = [[0, 0], [1, 0], [0, 1], [1, 1]]
test_label = clf.predict(test_data)

# テスト結果を評価する,accuracy_score()
print("予測対象：", test_data, ", 予測結果→", test_label)
print("正解率＝", accuracy_score([0, 1, 1, 0], test_label))