import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder

# 假设我们有一组训练数据 X 和对应的目标变量 y

data = pd.read_excel('data_pro.xlsx')
X = data.iloc[:, :8].values.astype(float)
y = data.iloc[:, -1].values.astype(float)
label_encoder = LabelEncoder()
# 将特征值和对应的目标变量转换为 numpy 数组
X = np.array(X)
y = label_encoder.fit_transform(y)

# 创建选择 K 个最佳特征的好工具对象
k_best_features = SelectKBest(chi2, k=4)  # 在这个例子中，我们选择 4 个最佳特征值

# 使用卡方检验来选择 K 个最佳特征值并进行拟合
k_best_features.fit_transform(X, y)

# 获取得分和 p 值，并按重要性降序排列特征
best_feature_scores = k_best_features.scores_
best_feature_p_values = k_best_features.pvalues_
best_feature_indices = np.argsort(best_feature_scores)[::-1]
'''
# 输出重要特征值及其分数和 p 值
for i in range(len(best_feature_indices)):
    print("Feature", i+1, ":", best_feature_indices[i],
          " Score:", best_feature_scores[best_feature_indices[i]],
          " p-value:", best_feature_p_values[best_feature_indices[i]])
'''

plt.scatter(best_feature_scores, best_feature_p_values)
plt.xlabel('X2')
plt.ylabel('P-value')
plt.title('relationship')
plt.show()
