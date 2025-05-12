from django.db import models

class GlassProcessingSalesSingleCompany(models.Model):
    """
    单公司玻璃加工销售数据模型
    对应glass_processing_sales_single_company表
    """
    company_name = models.CharField('公司名称', max_length=100)
    region = models.CharField('地区', max_length=100)
    product_name = models.CharField('产品名称', max_length=50)
    quantity = models.IntegerField('数量')
    original_price = models.DecimalField('原始价格', max_digits=10, decimal_places=2)
    sale_price = models.DecimalField('销售价格', max_digits=10, decimal_places=2)
    sales_amount = models.DecimalField('销售金额', max_digits=12, decimal_places=2)
    net_margin = models.DecimalField('净利润率', max_digits=5, decimal_places=2)
    sale_date = models.DateField('销售日期')
    
    class Meta:
        verbose_name = '单公司玻璃销售数据'
        verbose_name_plural = verbose_name
        db_table = 'glass_processing_sales_single_company'  # 指定数据库表名
        
    def __str__(self):
        return f"{self.company_name}-{self.product_name}"