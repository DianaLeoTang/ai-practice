import pandas as pd # 导入Pandas数据处理工具
import matplotlib as mpl
import matplotlib.pyplot as plt
# 设置字体路径
font_path = '/System/Library/AssetsV2/com_apple_MobileAsset_Font7/aa99d0b2bad7f797f38b49d46cde28fd4b58876e.asset/AssetData/Xingkai.ttc'
font_prop = mpl.font_manager.FontProperties(fname=font_path)
mpl.rcParams['font.family'] = font_prop.get_name() 

# 读取数据
data = pd.read_csv('某地乳腺检查数据.csv')
data.head()  # head方法显示前5行数据
# 设置X和y
y = data["诊断结果"]
X = data.drop(["诊断结果", "ID"], axis=1)
X

# 设置字体为SimHei，以正常显示中文标签
plt.rcParams["font.family"]=font_prop.get_name() 
plt.rcParams['font.sans-serif']=font_prop.get_name() 
# 用来正常显示负号
plt.rcParams['axes.unicode_minus']=False 

# 显示y的柱状图
y.value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('诊断结果分布')
plt.xlabel('诊断结果')
plt.ylabel('数量')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# 显示特征数据集的统计信息
X.describe()
# 查看X的特征名称
features = X.columns
features
# 绘制前三个特征的直方图
first_three_features = features[:3]

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10))

for i, feature in enumerate(first_three_features):
    axes[i].hist(X[feature], bins=30, color='skyblue', edgecolor='black')
    axes[i].set_title(f"{feature} - 直方图")
    axes[i].set_xlabel('值')
    axes[i].set_ylabel('频数')
    axes[i].grid(True, axis='y', linestyle='--', linewidth=0.7, alpha=0.6)

plt.tight_layout()
plt.show()
# 选择前10个特征
selected_features = features[:10]

# 绘制箱线图
plt.figure(figsize=(12, 8))
plt.boxplot([X[feature] for feature in selected_features], vert=True)
plt.xticks(range(1, len(selected_features) + 1), selected_features, rotation=45)
plt.title('箱线图 - 前10个特征')
plt.ylabel('值')
plt.grid(True, axis='y', linestyle='--', linewidth=0.7, alpha=0.6)
plt.tight_layout()
plt.show()
from sklearn.preprocessing import StandardScaler

# Setting X and y
y = data["诊断结果"]
X = data.drop(["诊断结果", "ID"], axis=1)

# Selecting the first 10 features
selected_features = X.columns[:10]

# Standardizing X
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Extracting the standardized values of the selected features
X_selected_standardized = X_standardized[:, :10]

# Plotting the boxplot for the standardized values of the selected features
import matplotlib.pyplot as plt

# Setting the font for displaying Chinese characters in the plot
plt.rcParams["font.family"]=font_prop.get_name()
plt.rcParams['font.sans-serif']=font_prop.get_name()
plt.rcParams['axes.unicode_minus']=False

plt.figure(figsize=(12, 8))
plt.boxplot(X_selected_standardized, vert=True)
plt.xticks(range(1, len(selected_features) + 1), selected_features, rotation=45)
plt.title('箱线图 - 标准化后的前10个特征')
plt.ylabel('标准化值')
plt.grid(True, axis='y', linestyle='--', linewidth=0.7, alpha=0.6)
plt.tight_layout()
plt.show()

import seaborn as sns # 导入Seaborn
# 绘制小提琴图
plt.figure(figsize=(12, 8))
sns.violinplot(data=pd.DataFrame(X_selected_standardized, columns=selected_features), palette="Set3")
plt.title('小提琴图 - 标准化后的前10个特征')
plt.ylabel('标准化值')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# 绘制相关性热图
correlation_matrix = pd.DataFrame(X_selected_standardized, columns=selected_features).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('前10个特征的相关性热图')
plt.tight_layout()
plt.show()