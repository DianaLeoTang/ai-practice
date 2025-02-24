'''
Author: Diana Tang
Date: 2025-02-23 23:08:10
LastEditors: Diana Tang
Description: some description
FilePath: /ai-practice/basic_part/直播带货AI实战5部曲.py
'''
# 导入数据可视化所需要的库
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘制图表
import seaborn as sns  # 导入Seaborn库，用于更高级的统计图表绘制
import matplotlib as mpl
# 设置字体路径
font_path = '/System/Library/AssetsV2/com_apple_MobileAsset_Font7/aa99d0b2bad7f797f38b49d46cde28fd4b58876e.asset/AssetData/Xingkai.ttc'

# 导入最基本的数据处理工具
import pandas as pd  # 导入Pandas库，用于数据处理和分析
df_ads = pd.read_csv('直播带货.csv')  # 读取名为'直播带货.csv'的数据文件，并将其存储在DataFrame对象df_ads中
df_ads.head(10)  # 显示DataFrame的前10行数据，用于初步查看数据内容



# 设置字体为Songti，以正常显示中文标签
# 将字体添加到 Matplotlib 的字体管理器中
font_prop = mpl.font_manager.FontProperties(fname=font_path)
mpl.rcParams['font.family'] = font_prop.get_name() 
plt.rcParams['font.sans-serif'] = font_prop.get_name() # 设置无衬线字体为Songti，确保中文标签正常显示
plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示负号，避免负号显示为方块

# 用matplotlib.pyplot的plot方法显示散点图
plt.plot(df_ads['转发量'], df_ads['成交额'], 'r.', label='数据点')  # 绘制转发量与成交额的散点图，红色点表示数据点
plt.xlabel('转发量')  # 设置x轴标签为“转发量”
plt.ylabel('成交额')  # 设置y轴标签为“成交额”
plt.legend()  # 显示图例，说明图中的数据点含义
plt.show()  # 显示绘制的图表

# 特征集，Drop掉标签字段
X = df_ads.drop(['成交额'], axis=1)  # 从数据集中删除“成交额”列，得到特征集X
y = df_ads.成交额  # 将“成交额”列作为标签集y
X.head()  # 显示特征集X的前几行数据
y.head()  # 显示标签集y的前几行数据

# 将数据集进行80%（训练集）和20%（验证集）的分割
from sklearn.model_selection import train_test_split  # 导入train_test_split函数，用于数据集分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  # 将数据集分为训练集和测试集，测试集占20%

# 导入线性回归算法模型
from sklearn.linear_model import LinearRegression  # 导入LinearRegression类，用于线性回归建模
model = LinearRegression()  # 创建线性回归模型对象
model.fit(X_train, y_train)  # 使用训练集数据训练模型，拟合函数并确定参数
y_pred = model.predict(X_test)  # 使用训练好的模型对测试集进行预测，得到预测值y_pred

# 测试集特征数据
df_ads_pred = X_test.copy()  # 复制测试集特征数据到df_ads_pred中
df_ads_pred['成交额真值'] = y_test  # 在df_ads_pred中添加一列，存储测试集的真实标签值
df_ads_pred['成交额预测值'] = y_pred  # 在df_ads_pred中添加一列，存储测试集的预测标签值
df_ads_pred  # 显示包含真实值和预测值的数据

# 评估模型
print("线性回归预测集评分：", model.score(X_test, y_test))  # 计算并打印模型在测试集上的评分（R²值）
print("线性回归训练集评分：", model.score(X_train, y_train))  # 计算并打印模型在训练集上的评分（R²值）

# 分离特征和标签
X = df_ads[['转发量']]  # 仅选择“转发量”列作为特征集X
y = df_ads.成交额  # 将“成交额”列作为标签集y

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  # 将数据集分为训练集和测试集，测试集占20%

# 使用线性回归模型
model = LinearRegression()  # 创建线性回归模型对象
model.fit(X_train, y_train)  # 使用训练集数据训练模型

# 预测
y_pred = model.predict(X_test)  # 使用训练好的模型对测试集进行预测，得到预测值y_pred

# 绘制预测直线
plt.figure(figsize=(10, 6))  # 创建一个新的图表，设置图表大小为10x6
plt.scatter(X_test, y_test, color='blue', label='真实值')  # 绘制测试集的真实值散点图，蓝色点表示
plt.plot(X_test, y_pred, color='red', linewidth=2, label='预测直线')  # 绘制预测值的回归直线，红色线表示
plt.xlabel('转发量')  # 设置x轴标签为“转发量”
plt.ylabel('成交额')  # 设置y轴标签为“成交额”
plt.title('转发量 vs 成交额')  # 设置图表标题为“转发量 vs 成交额”
plt.legend()  # 显示图例，说明图中的真实值和预测直线
plt.grid(True)  # 显示网格线，使图表更易读
plt.show()  # 显示绘制的图表