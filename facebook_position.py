import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 1. 获取数据
data = pd.read_csv(r".\data\FBlocation/train.csv")
# print(data.describe)

# 2.1 太大了，缩小数据范围
facebook_data = data.query("x>1.0&x<1.25&y>2.5&y<2.75")
# print(facebook_data.shape)

# 2.2 把时间戳转换为day,hour,weekday 等更有效的信息
time = pd.to_datetime(facebook_data["time"], unit="s")## 时间戳转换为yyyy-MM-dd hh:mm:ss 的函数
time = pd.DatetimeIndex(time)  # 时间转换为字符串
facebook_data.loc[:, "hour"] = time.hour
facebook_data.loc[:, "day"] = time.day
facebook_data.loc[:, "weekday"] = time.weekday
# print(facebook_data.head())

# 2.3 筛选出签到多的地方
place_count = facebook_data.groupby("place_id").count() ## 统计地点出现次数
# print(place_count.head())
place_count = place_count[place_count["row_id"] > 3]# 出现次数少的就去掉
facebook_data = facebook_data[facebook_data["place_id"].isin(place_count.index)]
# print(facebook_data.head())

# 2.4 确定特征值、目标值
x = facebook_data[["x", "y", "accuracy", "hour", "day", "weekday"]]
# print(x.head())
y = facebook_data["place_id"]
# print(y.head())
# 2.5 分割数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 3. 特征工程
# 3.1 数据标准化
# 3.1.1
transfer = StandardScaler()
# 3.1.2调用fit_trainsform 先计算均值，标准差，在进行标准化
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)
print("完成标准化")
# print(x_train)
# 4. 机器学习（knn + cv）
# 4.1 实例化估计器
estimator = KNeighborsClassifier()
# 4.2 调用交叉验证，网格搜索
param_grid = {"n_neighbors": [3, 5, 7]}
estimator = GridSearchCV(estimator=estimator,
                         param_grid=param_grid,
                         cv=5,
                         n_jobs=2)
print("完成交叉验证分隔")
# 4.3 训练
estimator.fit(x_train, y_train)
print("完成训练")
# 5. 模型评估
# 5.1 预测值输出
y_pre = estimator.predict(x_test)
print("预测值:", y_pre)
print("完成预测")
# 5.2 score
score = estimator.score(x_test, y_test)
print("准确率:", score)
best_estimator = estimator.best_estimator_
print("最好的模型是:", best_estimator)
best_score = estimator.best_score_
print("最高的准确率:", best_score)
# result = estimator.cv_results_
# print("整体结果：", result)

