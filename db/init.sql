use glass_analysis;

CREATE TABLE glass_processing_sales_large (
    id int PRIMARY KEY AUTO_INCREMENT,
    order_id VARCHAR(20),
    customer_id VARCHAR(20),
    product_name VARCHAR(50),
    quantity INT,
    unit_price DECIMAL(10,2),
    sale_date DATE,
    region VARCHAR(50),
    total_amount DECIMAL(12,2)
);

CREATE TABLE glass_processing_sales_single_company (
    id int PRIMARY KEY AUTO_INCREMENT,
    company_name VARCHAR(100),
    region VARCHAR(100),
    product_name VARCHAR(50),
    quantity INT,
    original_price DECIMAL(10,2),
    sale_price DECIMAL(10,2),
    sales_amount DECIMAL(12,2),
    net_margin DECIMAL(5,2),
    sale_date DATE
);

--
-- -- 用户表（Django内置）
-- CREATE TABLE `auth_user` (
--   `id` int NOT NULL AUTO_INCREMENT,
--   `password` varchar(128) NOT NULL,
--   `last_login` datetime(6) DEFAULT NULL,
--   `is_superuser` tinyint(1) NOT NULL,
--   `username` varchar(150) NOT NULL,
--   `first_name` varchar(150) NOT NULL,
--   `last_name` varchar(150) NOT NULL,
--   `email` varchar(254) NOT NULL,
--   `is_staff` tinyint(1) NOT NULL,
--   `is_active` tinyint(1) NOT NULL,
--   `date_joined` datetime(6) NOT NULL,
--   PRIMARY KEY (`id`),
--   UNIQUE KEY `username` (`username`)
-- ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
--
-- -- 用户配置文件表
-- CREATE TABLE `sales_profile` (
--   `id` bigint NOT NULL AUTO_INCREMENT,
--   `department` varchar(50) NOT NULL,
--   `position` varchar(50) NOT NULL,
--   `phone` varchar(20) NOT NULL,
--   `user_id` int NOT NULL,
--   PRIMARY KEY (`id`),
--   UNIQUE KEY `user_id` (`user_id`),
--   CONSTRAINT `sales_profile_user_id_fk` FOREIGN KEY (`user_id`) REFERENCES `auth_user` (`id`) ON DELETE CASCADE
-- ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
--
-- -- 创建初始管理员用户
-- -- 密码为 admin，使用Django的PBKDF2算法加密
-- INSERT INTO auth_user (password, is_superuser, username, first_name, last_name, email, is_staff, is_active, date_joined, last_login) VALUES
-- ('pbkdf2_sha256$870000$EfrSvkTtUx7XHIu9ardB4o$/TXQp6U1RYgOujvKValFHAwT31146QIDIJbhP4DOGaY=', 1, 'admin', '管理员', '', 'admin@example.com', 1, 1, NOW(), NOW());
--
--
-- -- 创建管理员用户的配置文件
-- INSERT INTO sales_profile (department, position, phone, user_id) VALUES
-- ('管理部', '系统管理员', '', LAST_INSERT_ID());