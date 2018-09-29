from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score  # 交叉验证
from sklearn.linear_model import Ridge  # 岭回归模型
import matplotlib.pylab as plt  # 可视化

alpha = 0.5
while alpha < 20:
    model = Ridge(alpha=alpha)
    X, Y = datasets.make_regression(n_samples=100, n_features=1, noise=10)
    scores = cross_val_score(estimator=model, X=X, y=Y, cv=5)  # 交叉验证 分成五组训练集、测试集
    print(scores.mean())
    print("alpha="+str(alpha))
    alpha *= 2
