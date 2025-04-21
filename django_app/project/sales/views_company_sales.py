import json
from decimal import Decimal

import pandas as pd
import numpy as np
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sales.models.models_company_sales import GlassProcessingSalesSingleCompany


# 假设直接从数据库获取数据
def get_company_sales_data():
    """
    从数据库获取单公司销售数据
    实际项目中应该使用ORM查询
    """
    # 这里应该使用Django ORM查询
    # 例如: GlassProcessingSalesSingleCompany.objects.all()
    # 但由于我们直接使用SQL文件，这里模拟从数据库获取数据
    try:
        # 尝试从CSV文件读取数据（实际项目中应从数据库读取）
        # 使用ORM查询所有数据
        queryset = GlassProcessingSalesSingleCompany.objects.all()

        # 转换为DataFrame
        df = pd.DataFrame.from_records(queryset.values(
            'company_name',
            'region',
            'product_name',
            'quantity',
            'original_price',
            'sale_price',
            'sales_amount',
            'net_margin',
            'sale_date'
        ))

        # 类型转换（如果需要）
        # if not df.empty:
        #     df['sale_date'] = pd.to_datetime(df['sale_date'])

        # df = pd.read_csv('db/glass_processing_sales_data_single_company.csv')
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        # 如果无法读取，返回空DataFrame
        return pd.DataFrame()

@login_required
def company_sales_analysis(request):
    """
    单公司销售数据分析视图
    """
    # 获取销售数据
    df = get_company_sales_data()
    print(df.head())

    # 如果数据为空，返回错误信息
    if df.empty:
        return render(request, 'sales/company_sales_analysis.html', {
            'error': '无法加载公司销售数据，请确保数据已正确导入。'
        })
    # 数据预处理
    # 确保日期格式正确
    if 'sale_date' in df.columns:
        df['sale_date'] = pd.to_datetime(df['sale_date'])
        df['month'] = df['sale_date'].dt.month
        df['year'] = df['sale_date'].dt.year

    # 1. 公司销售概览
    company_overview = df.groupby('company_name').agg({
        'sales_amount': 'sum',
        'quantity': 'sum',
        'net_margin': lambda x: (x * df.loc[x.index, 'sales_amount']).sum() / df.loc[x.index, 'sales_amount'].sum()
    }).reset_index()
    company_overview.columns = ['company_name', 'total_sales', 'total_quantity', 'avg_margin']
    company_overview = company_overview.sort_values('total_sales', ascending=False)

    # 2. 按地区分析
    region_analysis = df.groupby(['company_name', 'region']).agg({
        'sales_amount': 'sum'
    }).reset_index()

    # 获取销售额最高的公司
    top_company = company_overview.iloc[0]['company_name'] if not company_overview.empty else None

    # 3. 产品分析
    product_analysis = df.groupby(['company_name', 'product_name']).agg({
        'quantity': 'sum',
        'sales_amount': 'sum',
        'net_margin': 'mean'
    }).reset_index()

    # 4. 时间序列分析和预测
    if 'month' in df.columns and 'year' in df.columns and top_company:
        # 筛选出销售额最高的公司数据
        top_company_data = df[df['company_name'] == top_company]
        time_series = top_company_data.groupby(['year', 'month'])['sales_amount'].sum().reset_index()

        # 按时间排序并排除最新月份
        time_series = time_series.sort_values(['year', 'month'])
        if not time_series.empty:
            time_series = time_series.iloc[:-1]  # 移除最后一个月
        time_series['date_label'] = time_series.apply(lambda x: f"{int(x['year'])}-{int(x['month'])}", axis=1)

        # 创建产品类型特定的时间序列数据
        product_types = ['玻璃打磨', '玻璃雕刻', '玻璃抛光', '玻璃切割']
        product_time_series = {}
        product_predictions = {}
        
        # 为每种产品创建时间序列数据
        for product in product_types:
            product_data = top_company_data[top_company_data['product_name'] == product]
            if not product_data.empty:
                p_time_series = product_data.groupby(['year', 'month'])['sales_amount'].sum().reset_index()
                p_time_series = p_time_series.sort_values(['year', 'month'])
                if len(p_time_series) > 1:  # 确保有足够的数据
                    p_time_series = p_time_series.iloc[:-1]  # 移除最后一个月
                p_time_series['date_label'] = p_time_series.apply(lambda x: f"{int(x['year'])}-{int(x['month'])}", axis=1)
                product_time_series[product] = p_time_series

        # 销售预测 - 使用三级级联结构模型
        if len(time_series) >= 3:  # 至少需要3个数据点
            # 创建时间特征
            time_series['time_index'] = range(len(time_series))
            
            # 数据标准化 (min-max标准化)
            scaler = MinMaxScaler()
            sales_scaled = scaler.fit_transform(time_series[['sales_amount']])
            time_series['sales_amount_scaled'] = sales_scaled
            
            # 准备特征和目标变量
            X = time_series[['time_index']]
            y = time_series['sales_amount']
            
            # 1. 基模型预测层 - 训练三个基础模型
            # 随机森林回归器
            rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
            rf_model.fit(X, y)
            rf_pred = rf_model.predict(X)
            
            # XGBoost回归器
            xgb_model = XGBRegressor(learning_rate=0.05, n_estimators=200, random_state=42)
            xgb_model.fit(X, y)
            xgb_pred = xgb_model.predict(X)
            
            # 梯度提升树回归器
            gbdt_model = GradientBoostingRegressor(learning_rate=0.05, n_estimators=200, random_state=42)
            gbdt_model.fit(X, y)
            gbdt_pred = gbdt_model.predict(X)
            
            # 2. 特征增强层 - 将原始特征与基模型预测结果拼接
            enhanced_features = np.column_stack((X, rf_pred.reshape(-1, 1), xgb_pred.reshape(-1, 1), gbdt_pred.reshape(-1, 1)))
            
            # 3. 元模型决策层 - 使用随机森林作为元模型
            meta_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            meta_model.fit(enhanced_features, y)
            
            # 预测未来3个月
            future_indices = np.array(range(len(time_series), len(time_series) + 3)).reshape(-1, 1)
            
            # 使用基模型进行预测
            rf_future = rf_model.predict(future_indices)
            xgb_future = xgb_model.predict(future_indices)
            gbdt_future = gbdt_model.predict(future_indices)
            
            # 构建增强特征
            future_enhanced = np.column_stack((future_indices, rf_future.reshape(-1, 1), 
                                             xgb_future.reshape(-1, 1), gbdt_future.reshape(-1, 1)))
            
            # 使用元模型进行最终预测
            future_sales = meta_model.predict(future_enhanced)
            
            # 生成未来月份标签
            last_date = pd.to_datetime(f"{int(time_series.iloc[-1]['year'])}-{int(time_series.iloc[-1]['month'])}-01")
            future_dates = []
            for i in range(1, 4):
                next_date = last_date + pd.DateOffset(months=i)
                future_dates.append(f"{next_date.year}-{next_date.month}")

            # 计算模型评价指标
            y_pred = meta_model.predict(enhanced_features)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            mape = np.mean(np.abs((y.astype(float) - y_pred.astype(float)) / y.astype(float))) * 100

            # 确保数据为numpy数组且维度正确
            # 显式转换为浮点类型并确保维度一致
            y = np.array(y, dtype=float).flatten()
            y_pred = np.array(y_pred, dtype=float).flatten()
            
            # 异常处理
            try:
                if len(y) != len(y_pred):
                    corr = 0.0
                else:
                    corr_matrix = np.corrcoef(y, y_pred)
                    corr = corr_matrix[0, 1] if corr_matrix.size >= 4 else 0.0
            except Exception as e:
                print(f"计算相关系数时出错: {str(e)}")
                corr = 0.0

            # 预测结果
            predictions = {
                'company': top_company,
                'dates': future_dates,
                'values': future_sales.tolist(),
                'metrics': {
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'mape': float(mape),
                    'corr': float(corr)
                }
            }
            
            # 为每种产品类型进行预测
            product_types = df['product_name'].unique()[:3]  # 取销量前三的产品
            for product in product_types:
                product_data = top_company_data[top_company_data['product_name'] == product]
                
            if not product_data.empty:
                p_time_series = product_data.groupby(['year', 'month'])['sales_amount'].sum().reset_index()
                p_time_series = p_time_series.sort_values(['year', 'month'])
                if len(p_time_series) > 1:  # 确保有足够的数据
                    p_time_series = p_time_series.iloc[:-1]  # 移除最后一个月
                p_time_series['date_label'] = p_time_series.apply(lambda x: f"{int(x['year'])}-{int(x['month'])}", axis=1)
                product_time_series[product] = p_time_series

            # 销售预测 - 使用三级级联结构模型
            if len(time_series) >= 3:  # 至少需要3个数据点
                # 创建时间特征
                time_series['time_index'] = range(len(time_series))
                
                # 数据标准化 (min-max标准化)
                scaler = MinMaxScaler()
                sales_scaled = scaler.fit_transform(time_series[['sales_amount']])
                time_series['sales_amount_scaled'] = sales_scaled
                
                # 准备特征和目标变量
                X = time_series[['time_index']]
                y = time_series['sales_amount']
                
                # 1. 基模型预测层 - 训练三个基础模型
                # 随机森林回归器
                rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
                rf_model.fit(X, y)
                rf_pred = rf_model.predict(X)
                
                # XGBoost回归器
                xgb_model = XGBRegressor(learning_rate=0.05, n_estimators=200, random_state=42)
                xgb_model.fit(X, y)
                xgb_pred = xgb_model.predict(X)
                
                # 梯度提升树回归器
                gbdt_model = GradientBoostingRegressor(learning_rate=0.05, n_estimators=200, random_state=42)
                gbdt_model.fit(X, y)
                gbdt_pred = gbdt_model.predict(X)
                
                # 2. 特征增强层 - 将原始特征与基模型预测结果拼接
                enhanced_features = np.column_stack((X, rf_pred.reshape(-1, 1), xgb_pred.reshape(-1, 1), gbdt_pred.reshape(-1, 1)))
                
                # 3. 元模型决策层 - 使用随机森林作为元模型
                meta_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                meta_model.fit(enhanced_features, y)
                
                # 预测未来3个月
                future_indices = np.array(range(len(time_series), len(time_series) + 3)).reshape(-1, 1)
                
                # 使用基模型进行预测
                rf_future = rf_model.predict(future_indices)
                xgb_future = xgb_model.predict(future_indices)
                gbdt_future = gbdt_model.predict(future_indices)
                
                # 构建增强特征
                future_enhanced = np.column_stack((future_indices, rf_future.reshape(-1, 1), 
                                              xgb_future.reshape(-1, 1), gbdt_future.reshape(-1, 1)))
                
                # 使用元模型进行最终预测
                future_sales = meta_model.predict(future_enhanced)
                
                # 生成未来月份标签
                last_date = pd.to_datetime(f"{int(time_series.iloc[-1]['year'])}-{int(time_series.iloc[-1]['month'])}-01")
                future_dates = []
                for i in range(1, 4):
                    next_date = last_date + pd.DateOffset(months=i)
                    future_dates.append(f"{next_date.year}-{next_date.month}")

                # 计算模型评价指标
                y_pred = meta_model.predict(enhanced_features)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                mae = mean_absolute_error(y, y_pred)
                mape = np.mean(np.abs((y.astype(float) - y_pred.astype(float)) / y.astype(float))) * 100

                # 确保数据为numpy数组且维度正确
                # 显式转换为浮点类型并确保维度一致
                y = np.array(y, dtype=float).flatten()
                y_pred = np.array(y_pred, dtype=float).flatten()
                
                # 异常处理
                try:
                    if len(y) != len(y_pred):
                        corr = 0.0
                    else:
                        corr_matrix = np.corrcoef(y, y_pred)
                        corr = corr_matrix[0, 1] if corr_matrix.size >= 4 else 0.0
                except Exception as e:
                    print(f"计算相关系数时出错: {str(e)}")
                    corr = 0.0

                # 预测结果
                predictions = {
                    'company': top_company,
                    'dates': future_dates,
                    'values': future_sales.tolist(),
                    'metrics': {
                        'rmse': float(rmse),
                        'mae': float(mae),
                        'mape': float(mape),
                        'corr': float(corr)
                    }
                }
                
                # 为每种产品类型进行预测
                for product, p_time_series in product_time_series.items():
                    if len(p_time_series) >= 3:  # 确保有足够的数据进行预测
                        p_time_series['time_index'] = range(len(p_time_series))
                        
                        # 准备特征和目标变量
                        p_X = p_time_series[['time_index']]
                        p_y = p_time_series['sales_amount']
                        
                        # 基模型预测
                        p_rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
                        p_rf_model.fit(p_X, p_y)
                        p_rf_pred = p_rf_model.predict(p_X)
                        
                        p_xgb_model = XGBRegressor(learning_rate=0.05, n_estimators=200, random_state=42)
                        p_xgb_model.fit(p_X, p_y)
                        p_xgb_pred = p_xgb_model.predict(p_X)
                        
                        p_gbdt_model = GradientBoostingRegressor(learning_rate=0.05, n_estimators=200, random_state=42)
                        p_gbdt_model.fit(p_X, p_y)
                        p_gbdt_pred = p_gbdt_model.predict(p_X)
                        
                        # 特征增强
                        p_enhanced = np.column_stack((p_X, p_rf_pred.reshape(-1, 1), 
                                                  p_xgb_pred.reshape(-1, 1), p_gbdt_pred.reshape(-1, 1)))
                        
                        # 元模型
                        p_meta_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                        p_meta_model.fit(p_enhanced, p_y)
                        
                        # 预测未来3个月
                        p_future_indices = np.array(range(len(p_time_series), len(p_time_series) + 3)).reshape(-1, 1)
                        
                        # 基模型预测
                        p_rf_future = p_rf_model.predict(p_future_indices)
                        p_xgb_future = p_xgb_model.predict(p_future_indices)
                        p_gbdt_future = p_gbdt_model.predict(p_future_indices)
                        
                        # 构建增强特征
                        p_future_enhanced = np.column_stack((p_future_indices, p_rf_future.reshape(-1, 1), 
                                                        p_xgb_future.reshape(-1, 1), p_gbdt_future.reshape(-1, 1)))
                        
                        # 最终预测
                        p_future_sales = p_meta_model.predict(p_future_enhanced)
                        
                        # 计算评价指标
                        p_y_pred = p_meta_model.predict(p_enhanced)
                        p_rmse = np.sqrt(mean_squared_error(p_y, p_y_pred))
                        p_mae = mean_absolute_error(p_y, p_y_pred)
                        p_mape = np.mean(np.abs((p_y.astype(float) - p_y_pred.astype(float)) / p_y.astype(float)) * 100)
                        
                        # 确保产品预测数据维度正确
                        p_y = np.array(p_y).reshape(-1, 1)
                        p_y_pred = np.array(p_y_pred).reshape(-1, 1)




                        # if p_y.shape != p_y_pred.shape:
                        #     p_corr = 0.0
                        # else:
                        #     print(p_y,p_y_pred)
                        #     p_corr_matrix = np.corrcoef(p_y.flatten(), p_y_pred.flatten())
                        #     p_corr = p_corr_matrix[0, 1] if p_corr_matrix.size >= 4 else 0.0

                        # 显式转换为浮点类型并确保维度一致
                        y = np.array(y, dtype=float).flatten()
                        y_pred = np.array(y_pred, dtype=float).flatten()

                        # 异常处理
                        try:
                            if len(y) != len(y_pred):
                                corr = 0.0
                            else:
                                corr_matrix = np.corrcoef(y, y_pred)
                                corr = corr_matrix[0, 1] if corr_matrix.size >= 4 else 0.0
                        except Exception as e:
                            print(f"计算相关系数时出错: {str(e)}")
                            corr = 0.0


                        product_predictions[product] = {
                            'dates': future_dates,
                            'values': p_future_sales.tolist(),
                            'time_labels': p_time_series['date_label'].tolist(),
                            'time_values': p_time_series['sales_amount'].tolist(),
                            # 'metrics': metrics_data
                        }
        else:
            predictions = None
    else:
        time_series = pd.DataFrame()
        predictions = None
        product_predictions = {}

    # 5. 价格分析
    price_analysis = df.groupby(['company_name', 'product_name']).agg({
        'original_price': 'mean',
        'sale_price': 'mean'
    }).reset_index()
    price_analysis['discount_rate'] = (price_analysis['original_price'] - price_analysis['sale_price']) / price_analysis['original_price'] * 100

    # 准备图表数据
    chart_data = {
        'company_names': company_overview['company_name'].tolist(),
        'company_sales': company_overview['total_sales'].tolist(),
        'company_margins': company_overview['avg_margin'].tolist(),
        'time_labels': time_series['date_label'].tolist() if not time_series.empty else [],
        'time_values': time_series['sales_amount'].tolist() if not time_series.empty else [],
        'predictions': predictions,
        'product_predictions': product_predictions,
    }

    # 计算关键指标
    total_sales = df['sales_amount'].sum()
    avg_margin = (df['net_margin'] * df['sales_amount']).sum() / total_sales if total_sales > 0 else 0
    total_companies = df['company_name'].nunique()

    # 汇总统计数据
    summary_data = {
        'total_sales': total_sales,
        'avg_margin': avg_margin,
        'total_companies': total_companies,
        'top_company': top_company,
    }

    # 在返回render之前添加类型转换
    def convert_decimals(obj):
        if isinstance(obj, Decimal):
            return float(obj)
        raise TypeError

    chart_data = json.loads(json.dumps(chart_data, default=convert_decimals))
    summary_data = json.loads(json.dumps(summary_data, default=convert_decimals))


    return render(request, 'sales/company_sales_analysis.html', {
        'chart_data': chart_data,
        'summary_data': summary_data,
        'company_overview': company_overview.to_dict('records'),
        'product_analysis': product_analysis.to_dict('records'),
        'price_analysis': price_analysis.to_dict('records'),
    })