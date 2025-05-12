import os
import django
import MySQLdb
# from django.contrib.auth.hashers import make_password
# 加载环境变量（需提前创建.env文件）


# 数据库配置（从环境变量读取）
DB_CONFIG = {
    'host': 'db',
    'user': 'root',
    'password': '123456',
    'db': 'glass_analysis',
    'charset': 'utf8mb4'
}


def execute_sql_script():
    conn = None
    try:
        # 连接数据库
        conn = MySQLdb.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # 开启事务
        conn.begin()

        # 1. 创建用户配置表
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS `sales_profile` (
          `id` bigint NOT NULL AUTO_INCREMENT,
          `department` varchar(50) NOT NULL,
          `position` varchar(50) NOT NULL,
          `phone` varchar(20) NOT NULL,
          `user_id` int NOT NULL,
          PRIMARY KEY (`id`),
          UNIQUE KEY `user_id` (`user_id`),
          CONSTRAINT `sales_profile_user_id_fk` 
            FOREIGN KEY (`user_id`) 
            REFERENCES `auth_user` (`id`) 
            ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
        cursor.execute(create_table_sql)

        # 2. 动态生成安全密码
        raw_password = 'admin'  # 实际生产环境应从安全渠道获取
        hashed_password = "pbkdf2_sha256$870000$EfrSvkTtUx7XHIu9ardB4o$/TXQp6U1RYgOujvKValFHAwT31146QIDIJbhP4DOGaY="

        # 3. 插入管理员用户（如果不存在）
        check_user_sql = "SELECT id FROM auth_user WHERE username = 'admin'"
        cursor.execute(check_user_sql)
        user_exists = cursor.fetchone()

        if not user_exists:
            insert_user_sql = """
            INSERT INTO auth_user (
                password, is_superuser, username, 
                first_name, last_name, email, 
                is_staff, is_active, date_joined, last_login
            ) VALUES (
                %s, 1, 'admin', 
                '管理员', '', 'admin@example.com', 
                1, 1, NOW(), NOW()
            )
            """
            cursor.execute(insert_user_sql, (hashed_password,))
            user_id = cursor.lastrowid

            # 4. 插入用户配置
            insert_profile_sql = """
            INSERT INTO sales_profile 
                (department, position, phone, user_id)
            VALUES
                (%s, %s, %s, %s)
            """
            profile_data = ('管理部', '系统管理员', '', user_id)
            cursor.execute(insert_profile_sql, profile_data)

            print(f"创建管理员成功，用户ID: {user_id}")
        else:
            print("管理员用户已存在，跳过创建")

        # 提交事务
        conn.commit()

    except MySQLdb.Error as e:
        print(f"数据库操作失败: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    execute_sql_script()