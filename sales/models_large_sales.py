from django.db import models

class GlassProcessingSalesLarge(models.Model):
    """
    大型玻璃加工销售数据模型
    对应glass_processing_sales_large表
    """
    order_id = models.CharField('订单ID', max_length=20)
    customer_id = models.CharField('客户ID', max_length=20)
    product_name = models.CharField('产品名称', max_length=50)
    quantity = models.IntegerField('数量')
    unit_price = models.DecimalField('单价', max_digits=10, decimal_places=2)
    sale_date = models.DateField('销售日期')
    region = models.CharField('地区', max_length=50)
    total_amount = models.DecimalField('总金额', max_digits=12, decimal_places=2)
    
    class Meta:
        verbose_name = '大型玻璃销售数据'
        verbose_name_plural = verbose_name
        db_table = 'glass_processing_sales_large'  # 指定数据库表名
        
    def __str__(self):
        return f"{self.order_id}-{self.product_name}"