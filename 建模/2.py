# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import xgboost as xgb
# import statsmodels.api as sm
# from statsmodels.tsa.seasonal import seasonal_decompose
# from sklearn.metrics import mean_absolute_error
# from sklearn.model_selection import train_test_split
#
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
#
#
# # ================= 数据读取与预处理 =================
# def load_and_preprocess():
#     # 读取销售数据
#     sales_df = pd.read_csv("glass_processing_sales_data_single_company.csv",
#                            parse_dates=['销售日期'])
#
#     # 按产品和时间排序（确保每个产品内部按时间排序）
#     sales_df = sales_df.sort_values(['产品名称', '销售日期']).reset_index(drop=True)
#
#     # 创建时间特征
#     sales_df['year'] = sales_df['销售日期'].dt.year
#     sales_df['month'] = sales_df['销售日期'].dt.month
#     sales_df['week'] = sales_df['销售日期'].dt.isocalendar().week
#     sales_df['quarter'] = sales_df['销售日期'].dt.quarter
#
#     # 添加滞后特征（分组后严格按时间计算滞后）
#     for lag in [1, 2, 3]:
#         sales_df[f'价格滞后{lag}'] = sales_df.groupby('产品名称', group_keys=False)['销售价格'].apply(
#             lambda x: x.shift(lag)
#         )
#         sales_df[f'销量滞后{lag}'] = sales_df.groupby('产品名称', group_keys=False)['销售数量'].apply(
#             lambda x: x.shift(lag)
#         )
#
#     # 添加移动平均特征（分组计算）
#     sales_df['3月价格均值'] = sales_df.groupby('产品名称', group_keys=False)['销售价格'].transform(
#         lambda x: x.rolling(window=3, min_periods=1).mean()
#     )
#     sales_df['6月销量均值'] = sales_df.groupby('产品名称', group_keys=False)['销售数量'].transform(
#         lambda x: x.rolling(window=6, min_periods=1).mean()
#     )
#
#     # 删除因滞后产生的空值（保留部分移动平均）
#     sales_df = sales_df.dropna(subset=[f'价格滞后3', f'销量滞后3'])
#
#     return sales_df
#
#
# # ================= 特征工程 =================
# def create_features(df, target_col):
#     # 基础特征列
#     base_features = ['year', 'month', 'week', 'quarter',
#                      '价格滞后1', '价格滞后2', '价格滞后3',
#                      '销量滞后1', '销量滞后2', '销量滞后3',
#                      '3月价格均值', '6月销量均值']
#
#     # 产品类型编码（使用实际存在的产品类别）
#     products = df['产品名称'].unique()
#     product_dummies = pd.get_dummies(df['产品名称'], prefix='产品')
#
#     # 合并特征时确保列对齐
#     X = pd.concat([df[base_features], product_dummies], axis=1)
#     y = df[target_col]
#     return X, y
#
#
# # ================= 时间序列分析 =================
# def time_series_analysis(product_df, target_col):
#     # 如果数据为常数，则不做分解和模型训练
#     if product_df[target_col].nunique() == 1:
#         constant_value = product_df[target_col].iloc[0]
#         print(f"注意：目标变量 {target_col} 数据为常数，值为 {constant_value}，跳过时间序列分解。")
#         return None
#
#     # 分解时间序列（此处周期设置为4，仅为示例，实际可根据数据调整）
#     result = seasonal_decompose(product_df[target_col],
#                                 model='additive',
#                                 period=4)
#
#     # 绘制分解结果
#     fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
#     result.observed.plot(ax=ax1, title='原始序列')
#     result.trend.plot(ax=ax2, title='趋势成分')
#     result.seasonal.plot(ax=ax3, title='季节成分')
#     result.resid.plot(ax=ax4, title='残差成分')
#     plt.tight_layout()
#
#     # 训练SARIMA模型
#     model = sm.tsa.SARIMAX(product_df[target_col],
#                            order=(1, 1, 1),
#                            seasonal_order=(1, 1, 1, 12),
#                            enforce_stationarity=False)
#     sarima_results = model.fit()
#
#     # 模型诊断
#     sarima_results.plot_diagnostics(figsize=(15, 12))
#     plt.suptitle(f'{target_col} SARIMA模型诊断', y=1.02)
#
#     return sarima_results
#
#
# # ================= 模型训练与预测 =================
# def train_and_predict(product_df, target_col):
#     # 划分训练测试集
#     train_size = int(len(product_df) * 0.8)
#     train, test = product_df.iloc[:train_size], product_df.iloc[train_size:]
#
#     # 如果训练集目标变量为常数，则直接返回常数预测
#     if train[target_col].nunique() == 1:
#         constant_val = train[target_col].iloc[0]
#         print(f"注意：{target_col} 数据在训练集中为常数，值为 {constant_val}，直接返回常数预测。")
#         sarima_pred = pd.Series([constant_val] * len(test), index=test.index)
#         xgb_pred = np.array([constant_val] * len(test))
#         return sarima_pred, xgb_pred, test[target_col]
#
#     # 时间序列模型
#     sarima_model = sm.tsa.SARIMAX(train[target_col],
#                                   order=(1, 1, 1),
#                                   seasonal_order=(1, 1, 1, 12))
#     sarima_results = sarima_model.fit()
#
#     # XGBoost模型
#     X_train, y_train = create_features(train, target_col)
#     X_test, y_test = create_features(test, target_col)
#
#     xgb_model = xgb.XGBRegressor(
#         n_estimators=100,
#         max_depth=5,
#         learning_rate=0.1,
#         subsample=0.8,
#         colsample_bytree=0.8
#     )
#     xgb_model.fit(X_train, y_train)
#
#     # 生成预测
#     sarima_pred = sarima_results.get_forecast(steps=len(test)).predicted_mean
#     xgb_pred = xgb_model.predict(X_test)
#
#     return sarima_pred, xgb_pred, test[target_col]
#
#
# # ================= 模型融合 =================
# def kalman_filter_fusion(sarima_pred, xgb_pred, y_true):
#     # 初始化参数
#     P = np.ones(len(y_true))
#     Q = 0.1
#     R = 0.1
#     fused_pred = np.zeros(len(y_true))
#
#     for i in range(len(y_true)):
#         K = P[i] / (P[i] + R)
#         fused_pred[i] = sarima_pred[i] + K * (xgb_pred[i] - sarima_pred[i])
#         P[i] = P[i] - K * P[i] + Q
#
#     return fused_pred
#
#
# # ================= 可视化 =================
# def plot_results(y_true, sarima_pred, xgb_pred, fused_pred, title):
#     plt.figure(figsize=(15, 8))
#     plt.plot(y_true.index, y_true, label='实际值', marker='o')
#     plt.plot(y_true.index, sarima_pred, label='SARIMA预测', linestyle='--')
#     plt.plot(y_true.index, xgb_pred, label='XGBoost预测', linestyle='--')
#     plt.plot(y_true.index, fused_pred, label='融合预测', color='red', linewidth=2)
#
#     plt.title(f'{title}预测结果对比', fontsize=16)
#     plt.xlabel('日期', fontsize=12)
#     plt.ylabel(title, fontsize=12)
#     plt.legend()
#     plt.grid(True)
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()
#
#
# def data_quality_check(sales_df):
#     # 检查价格波动性
#     print("数据维度:", sales_df.shape)
#     print("前5行数据:\n", sales_df.head())
#
#     # 示例输出时间特征
#     print("时间特征示例:")
#     print(sales_df[['销售日期', 'year', 'month', 'week', 'quarter']].head())
#
#     # 验证滞后特征是否正确生成
#     print("价格滞后特征示例:")
#     # print(sales_df[['销售日期', '产品名称', '销售价格', '价格滞后1', '价格滞后2', '价格滞后3']].head(10))
#
#
# # ================= 主程序 =================
# if __name__ == "__main__":
#     # 数据预处理
#     sales_df = load_and_preprocess()
#     data_quality_check(sales_df)  # 新增数据质量检查
#
#     # 定义预测目标
#     targets = ['销售价格', '销售额', '净利率']
#
#     for target in targets:
#         print(f"\n=== 正在处理 {target} ===")
#
#         # 按产品循环
#         for product in sales_df['产品名称'].unique():
#             print(f"\n处理产品: {product}")
#             product_df = sales_df[sales_df['产品名称'] == product].set_index('销售日期')
#
#             # 时间序列分析（如果目标数据为常数，则此函数内部会提示并跳过分解）
#             _ = time_series_analysis(product_df, target)
#
#             # 训练与预测
#             sarima_pred, xgb_pred, y_test = train_and_predict(product_df, target)
#
#             # 模型融合
#             fused_pred = kalman_filter_fusion(sarima_pred.values, xgb_pred, y_test.values)
#
#             # 评估指标
#             print(f"\n{product} {target} 评估结果:")
#             print(f"SARIMA MAE: {mean_absolute_error(y_test, sarima_pred):.2f}")
#             print(f"XGBoost MAE: {mean_absolute_error(y_test, xgb_pred):.2f}")
#             print(f"融合模型 MAE: {mean_absolute_error(y_test, fused_pred):.2f}")
#
#             # 可视化结果
#             plot_results(y_test, sarima_pred, xgb_pred, fused_pred,
#                          f"{product} {target}")
#
#     # 展示所有图表
#     plt.show()
