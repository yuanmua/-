CREATE DATABASE glass_analysis CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
use glass_analysis;


-- 创建数据表（需先运行Django迁移生成）
-- 以下SQL结构应与Django模型完全一致
CREATE TABLE `sales_salesdata` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `month` varchar(10) NOT NULL,
  `product` varchar(50) NOT NULL,
  `sales_volume` double DEFAULT NULL,
  `sales_amount` double NOT NULL,
  `cost` double DEFAULT NULL,
  `gross_margin` double NOT NULL,
  `unit_price` double NOT NULL,
  `material_ratio` double NOT NULL,
  `major_client` varchar(100) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


INSERT INTO sales_salesdata (month, product, sales_volume, sales_amount, cost, gross_margin, unit_price, material_ratio, major_client) VALUES
('1月', '钢化玻璃', 3450, 269290, 128823, 0.5217, 78.06, 0.65, '麒龙集团（60%）'),
('1月', '夹胶玻璃', 1890, 1200621, 573390, 0.5223, 635.3, 0.72, '久高装饰（70%）'),
('2月', '钢化玻璃', 2180, 120000, 82013, 0.3165, 55.05, 0.58, '麒龙集团（50%）'),
('4月', '钢化玻璃', 2800, 180000, 100800, 0.4400, 64.29, 0.60, '久高装饰（60%）'),
('5月', '夹胶玻璃', 2500, 1500000, 720000, 0.5200, 600, 0.70, '麒龙集团（55%）'),
('6月', '钢化玻璃', 3200, 250000, 140800, 0.4368, 78.13, 0.62, '久高装饰（65%）'),
('7月', '夹胶玻璃', 2000, 1200000, 816000, 0.3200, 600, 0.75, '麒龙集团（70%）'),
('8月', '钢化玻璃', 3500, 245000, 137200, 0.4400, 70, 0.63, '久高装饰（58%）'),
('9月', '夹胶玻璃', 2800, 1680000, 806400, 0.5200, 600, 0.71, '麒龙集团（62%）'),
('10月', '钢化玻璃', 3000, 210000, 126000, 0.4000, 70, 0.64, '久高装饰（55%）'),
('11月', '夹胶玻璃', 2200, 1320000, 677600, 0.4864, 600, 0.73, '麒龙集团（68%）'),
('12月', '钢化玻璃', 3600, 252000, 140400, 0.4433, 70, 0.62, '久高装饰（60%）');