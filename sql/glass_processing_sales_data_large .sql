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