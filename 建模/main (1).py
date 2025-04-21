import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.seasonal import STL


# ======================
# 数据预处理与特征工程
# ======================
def preprocess_data(df):
    # 时间特征提取（数值化处理）
    df['hour'] = df.index.hour
    df['15min_block'] = df.index.minute // 15  # 改为数值特征
    df['is_peak'] = df.index.hour.isin([7, 8, 17, 18]).astype(int)

    # 扩展滞后特征（24步=2小时窗口）
    n_lags = 24
    for i in range(1, n_lags + 1):
        df[f'lag_{i}'] = df['vehicle_count'].shift(i)

    # 统计特征
    df['rolling_3h_mean'] = df['vehicle_count'].rolling(window=36).mean()
    df['rolling_6h_std'] = df['vehicle_count'].rolling(window=72).std()

    # 趋势分解
    stl = STL(df['vehicle_count'], period=288)
    result = stl.fit()
    df['detrended'] = result.resid + result.seasonal

    # 处理缺失值
    df.dropna(inplace=True)
    return df

def dynamic_rolling_predict(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    predictions = []
    X_test = X_test.copy()

    for i in range(len(X_test)):
        # 获取当前步特征
        current_features = X_test.iloc[i:i + 1]

        # 预测并记录
        pred = model.predict(current_features)
        predictions.append(pred[0])

        # 更新后续lag特征（仅当不是最后一步时）
        if i < len(X_test) - 1:
            # 前移lag特征
            for lag in range(1, 24):
                X_test.iloc[i + 1, X_test.columns.get_loc(f'lag_{lag}')] = X_test.iloc[i + 1][f'lag_{lag + 1}']
            # 更新最新lag
            X_test.iloc[i + 1, X_test.columns.get_loc('lag_24')] = pred[0]

    return np.array(predictions)

def build_ensemble_model():
    # 基模型配置
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.8,
        tree_method='hist'  # 提升分类特征处理能力
    )

    gbrt = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=7,
        random_state=42
    )

    # 堆叠集成
    return StackingRegressor(
        estimators=[('rf', rf), ('xgb', xgb), ('gbrt', gbrt)],
        final_estimator=RandomForestRegressor(n_estimators=100),
        passthrough=True
    )


# ======================
# 主执行流程
# ======================
if __name__ == "__main__":
    # 数据加载
    df = pd.read_excel('data413.xlsx',
                       sheet_name='plate413-a',
                       parse_dates=['watch_time'],
                       index_col='watch_time')
    df.rename(columns={'car_no_num': 'vehicle_count'}, inplace=True)

    # 预处理与特征检查
    df = preprocess_data(df)
    print("特征数据类型验证:")
    print(df.dtypes)

    # 分割特征与目标
    X = df.drop(columns=['vehicle_count'])
    y = df['vehicle_count']

    # 时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=3)
    metrics = {'MSE': [], 'MAE': [], 'R2': [], 'Phase_Lag': []}

    # 可视化初始化
    plt.figure(figsize=(15, 10))

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 模型训练与预测
        model = build_ensemble_model()
        y_pred = dynamic_rolling_predict(model, X_train, y_train, X_test)

        # 评估指标
        metrics['MSE'].append(mean_squared_error(y_test, y_pred))
        metrics['MAE'].append(mean_absolute_error(y_test, y_pred))
        metrics['R2'].append(r2_score(y_test, y_pred))
        cross_corr = np.correlate(y_test, y_pred, mode='full')
        metrics['Phase_Lag'].append(np.argmax(cross_corr) - len(y_test) + 1)

        # 可视化预测结果
        plt.subplot(3, 1, fold + 1)
        plt.plot(y_test.index, y_test, label='Actual', alpha=0.8, linewidth=1.5)
        plt.plot(y_test.index, y_pred, label='Predicted', linestyle='--', linewidth=1.2)
        plt.title(f'Fold {fold + 1} (Lag: {metrics["Phase_Lag"][-1]} steps)', fontsize=10)
        plt.grid(alpha=0.3)
        plt.legend()

    plt.tight_layout()

    # 残差分析
    residuals = y_test - y_pred[-len(y_test):]  # 对齐长度

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].scatter(y_pred, residuals, alpha=0.6, c='darkgreen', edgecolor='white')
    ax[0].axhline(0, color='red', linestyle='--')
    ax[0].set_title('Residual Distribution')

    ax[1].hist(residuals, bins=25, color='steelblue', edgecolor='black')
    ax[1].set_title('Residual Histogram')

    # 打印评估结果
    print(f'''
    ========== 最终评估 ==========
    平均MSE: {np.mean(metrics['MSE']):.1f} ± {np.std(metrics['MSE']):.1f}
    平均MAE: {np.mean(metrics['MAE']):.1f} ± {np.std(metrics['MAE']):.1f}
    平均R²: {np.mean(metrics['R2']):.2f} ± {np.std(metrics['R2']):.2f}
    平均相位差: {np.mean(metrics['Phase_Lag']):.1f} steps
    ''')

    plt.show()