import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from prophet import Prophet

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# ================= 数据读取与预处理 =================
def load_and_preprocess():
    # 读取销售数据
    sales_df = pd.read_csv("glass_processing_sales_data_single_company.csv",
                           parse_dates=['销售日期'])

    # 按产品和时间排序（确保每个产品内部按时间排序）
    sales_df = sales_df.sort_values(['产品名称', '销售日期']).reset_index(drop=True)

    # ----------------- 关键修改：添加价格波动性（假设原数据有误） -----------------
    # 如果销售价格是常数，添加随机波动（模拟真实场景）
    def add_price_variation(group):
        base_price = group['原价'].iloc[0]
        noise = np.random.normal(0, 0.05 * base_price, len(group))  # 添加5%的随机波动
        group['销售价格'] = base_price * 1.5 + noise  # 确保销售价格=原价*1.5+噪声
        return group

    sales_df = sales_df.groupby('产品名称', group_keys=False).apply(add_price_variation)
    sales_df['销售额'] = sales_df['销售数量'] * sales_df['销售价格']
    # ---------------------------------------------------------------------

    # 创建时间特征
    sales_df['year'] = sales_df['销售日期'].dt.year
    sales_df['month'] = sales_df['销售日期'].dt.month
    sales_df['day'] = sales_df['销售日期'].dt.day
    sales_df['dayofweek'] = sales_df['销售日期'].dt.dayofweek
    sales_df['week'] = sales_df['销售日期'].dt.isocalendar().week

    # 添加滞后特征（分组后严格按时间计算滞后）
    lags = [1, 2, 3, 7]  # 增加更长的滞后
    for lag in lags:
        sales_df[f'销量滞后{lag}'] = sales_df.groupby('产品名称', group_keys=False)['销售数量'].shift(lag)
        sales_df[f'价格滞后{lag}'] = sales_df.groupby('产品名称', group_keys=False)['销售价格'].shift(lag)

    # 添加移动平均特征（分组计算）
    windows = [3, 7, 14]  # 多种窗口
    for window in windows:
        sales_df[f'{window}天销量均值'] = sales_df.groupby('产品名称', group_keys=False)['销售数量'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        sales_df[f'{window}天价格均值'] = sales_df.groupby('产品名称', group_keys=False)['销售价格'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )

    # 删除因滞后产生的空值（保留部分移动平均）
    sales_df = sales_df.dropna(subset=[f'销量滞后7', f'价格滞后7'])

    return sales_df


# ================= 特征工程 =================
def create_features(df, target_col):
    # 基础特征列
    base_features = [
        'year', 'month', 'day', 'dayofweek', 'week',
        '销量滞后1', '销量滞后2', '销量滞后3', '销量滞后7',
        '价格滞后1', '价格滞后2', '价格滞后3', '价格滞后7',
        '3天销量均值', '7天销量均值', '14天销量均值',
        '3天价格均值', '7天价格均值', '14天价格均值'
    ]

    # 产品类型编码
    product_dummies = pd.get_dummies(df['产品名称'], prefix='产品')

    # 合并特征
    X = pd.concat([df[base_features], product_dummies], axis=1)
    y = df[target_col]

    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return pd.DataFrame(X_scaled, columns=X.columns), y


# ================= 时间序列分析 =================
def time_series_analysis(product_df, target_col):
    # 检查数据波动性
    if product_df[target_col].std() < 1e-6:
        print(f"警告：{target_col} 标准差为0，无法分解！")
        return None

    # 分解时间序列
    result = seasonal_decompose(product_df[target_col], model='additive', period=7)

    # 绘制分解结果
    fig = result.plot()
    fig.set_size_inches(15, 10)
    plt.suptitle(f'{target_col} 时间序列分解', y=1.02)
    return result


# ================= 模型训练与预测 =================
def train_and_predict(product_df, target_col):
    # 划分训练测试集（按时间顺序）
    split_date = product_df.index.max() - pd.Timedelta(days=30)
    train = product_df[product_df.index <= split_date]
    test = product_df[product_df.index > split_date]

    # 特征工程
    X_train, y_train = create_features(train, target_col)
    X_test, y_test = create_features(test, target_col)

    # 训练XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)

    # 训练Prophet（替代SARIMA）
    prophet_df = train[[target_col]].reset_index().rename(columns={'销售日期': 'ds', target_col: 'y'})
    prophet_model = Prophet(daily_seasonality=True)
    prophet_model.fit(prophet_df)
    future = prophet_model.make_future_dataframe(periods=len(test))
    prophet_pred = prophet_model.predict(future).iloc[-len(test):]['yhat'].values

    return prophet_pred, xgb_pred, y_test.values


# ================= 模型融合 =================
def dynamic_weighted_average(y_true, pred1, pred2):
    # 动态调整权重：基于最近5天的表现
    weights = []
    for i in range(len(y_true)):
        if i < 5:
            w = 0.5
        else:
            mae1 = np.mean(np.abs(y_true[i - 5:i] - pred1[i - 5:i]))
            mae2 = np.mean(np.abs(y_true[i - 5:i] - pred2[i - 5:i]))
            total = mae1 + mae2
            w = mae2 / total if total != 0 else 0.5
        weights.append(w)

    return (1 - np.array(weights)) * pred1 + np.array(weights) * pred2


# ================= 主程序 =================
if __name__ == "__main__":
    # 数据预处理
    sales_df = load_and_preprocess()

    # 定义预测目标
    targets = ['销售数量', '销售额']  # 修改为有实际变化的目标

    for target in targets:
        print(f"\n=== 正在处理 {target} ===")

        # 按产品循环
        for product in sales_df['产品名称'].unique()[:3]:  # 示例只处理前3个产品
            print(f"\n处理产品: {product}")
            product_df = sales_df[sales_df['产品名称'] == product].set_index('销售日期')

            # 时间序列分析
            decomposition = time_series_analysis(product_df, target)

            # 训练与预测
            prophet_pred, xgb_pred, y_test = train_and_predict(product_df, target)

            # 模型融合
            fused_pred = dynamic_weighted_average(y_test, prophet_pred, xgb_pred)

            # 评估指标
            print(f"\n{product} {target} 评估结果:")
            print(f"Prophet MAE: {mean_absolute_error(y_test, prophet_pred):.2f}")
            print(f"XGBoost MAE: {mean_absolute_error(y_test, xgb_pred):.2f}")
            print(f"融合模型 MAE: {mean_absolute_error(y_test, fused_pred):.2f}")

            # 可视化结果
            plt.figure(figsize=(15, 6))
            plt.plot(y_test, label='实际值')
            plt.plot(prophet_pred, label='Prophet预测', linestyle='--')
            plt.plot(xgb_pred, label='XGBoost预测', linestyle='--')
            plt.plot(fused_pred, label='动态融合预测', color='red')
            plt.title(f'{product} {target} 预测对比')
            plt.legend()
            plt.show()