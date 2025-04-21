CREATE DATABASE glass_analysis CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
use glass_analysis;

-- 创建数据表（需先运行Django迁移生成）
-- 以下SQL结构应与Django模型完全一致

-- 销售数据表
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

# -- 用户表（Django内置）
# CREATE TABLE `auth_user` (
#   `id` int NOT NULL AUTO_INCREMENT,
#   `password` varchar(128) NOT NULL,
#   `last_login` datetime(6) DEFAULT NULL,
#   `is_superuser` tinyint(1) NOT NULL,
#   `username` varchar(150) NOT NULL,
#   `first_name` varchar(150) NOT NULL,
#   `last_name` varchar(150) NOT NULL,
#   `email` varchar(254) NOT NULL,
#   `is_staff` tinyint(1) NOT NULL,
#   `is_active` tinyint(1) NOT NULL,
#   `date_joined` datetime(6) NOT NULL,
#   PRIMARY KEY (`id`),
#   UNIQUE KEY `username` (`username`)
# ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 用户配置文件表
CREATE TABLE `sales_profile` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `department` varchar(50) NOT NULL,
  `position` varchar(50) NOT NULL,
  `phone` varchar(20) NOT NULL,
  `user_id` int NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `user_id` (`user_id`),
  CONSTRAINT `sales_profile_user_id_fk` FOREIGN KEY (`user_id`) REFERENCES `auth_user` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- 销售数据记录
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

-- 创建初始管理员用户
-- 密码为 admin123，使用Django的PBKDF2算法加密
INSERT INTO auth_user (password, is_superuser, username, first_name, last_name, email, is_staff, is_active, date_joined, last_login) VALUES
('pbkdf2_sha256$260000$qMygN1PS6a7TGHE9fbnJRg$+5+2u1QBNSbr1uRGCs58WdZ5DAfyFQBIbdmPHMV0jK8=', 1, 'admin', '管理员', '', 'admin@example.com', 1, 1, NOW(), NOW());

-- 创建管理员用户的配置文件
INSERT INTO sales_profile (department, position, phone, user_id) VALUES
('管理部', '系统管理员', '', LAST_INSERT_ID());