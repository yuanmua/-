import pandas as pd
import pymysql as pymysql

from sqlalchemy import create_engine

# 创建数据库连接（需修改连接参数）
DB_CONFIG = {
    'user': 'root',
    'password': '123s123s',
    'host': 'localhost',
    'database': 'glass_analysis'
}
engine = create_engine(
    f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}")


# 处理第一个 CSV 文件
def process_large_sales():
    df = pd.read_csv("glass_processing_sales_data_large.csv")

    # 列名映射
    column_mapping = {
        '订单ID': 'order_id',
        '客户ID': 'customer_id',
        '产品名称': 'product_name',
        '销售数量': 'quantity',
        '单价': 'unit_price',
        '销售日期': 'sale_date',
        '销售区域': 'region',
        '总金额': 'total_amount'
    }
    df = df.rename(columns=column_mapping)

    # 写入数据库
    df.to_sql(
        name='glass_processing_sales_large',
        con=engine,
        if_exists='append',
        index=False
    )


# 处理第二个 CSV 文件
def process_single_company_sales():
    df = pd.read_csv("glass_processing_sales_data_single_company.csv")

    # 列名映射
    column_mapping = {
        '企业名称': 'company_name',
        '销售区域': 'region',
        '产品名称': 'product_name',
        '销售数量': 'quantity',
        '原价': 'original_price',
        '销售价格': 'sale_price',
        '销售额': 'sales_amount',
        '净利率': 'net_margin',
        '销售日期': 'sale_date'
    }
    df = df.rename(columns=column_mapping)

    # 写入数据库
    df.to_sql(
        name='glass_processing_sales_single_company',
        con=engine,
        if_exists='append',
        index=False
    )
# 执行插入操作
if __name__ == "__main__":
    process_large_sales()
    process_single_company_sales()
    print("数据插入完成")