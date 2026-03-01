DROP DATABASE IF EXISTS ph_provinces;
CREATE DATABASE ph_provinces;
USE ph_provinces;

CREATE TABLE region(
    region_id INT PRIMARY KEY NOT NULL,
    region_name VARCHAR(200)
);

CREATE TABLE province(
    province_id INT PRIMARY KEY NOT NULL,
    province_name VARCHAR(200),
    region_id INT NOT NULL, 
    FOREIGN KEY(region_id)
    REFERENCES ph_provinces.region(region_id)
);

CREATE TABLE municipality(
    municipality_id INT PRIMARY KEY NOT NULL,
    municipality_name VARCHAR(200),
    province_id INT NOT NULL, 
    FOREIGN KEY(province_id)
    REFERENCES ph_provinces.province(province_id)
);

CREATE TABLE barangay(
    barangay_id INT PRIMARY KEY NOT NULL, 
    barangay_name VARCHAR(200),
    municipality_id INT,
    FOREIGN KEY(municipality_id)
    REFERENCES ph_provinces.municipality(municipality_id)
);

LOAD DATA LOCAL INFILE 'C:\\ProgramData\\MySQL\\MySQL Server 8.0\\Uploads\\region_table.csv'
INTO TABLE region
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS
(region_id, region_name);

LOAD DATA LOCAL INFILE 'C:\\ProgramData\\MySQL\\MySQL Server 8.0\\Uploads\\province_table.csv'
INTO TABLE province
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS
(region_id, province_id, province_name);

LOAD DATA LOCAL INFILE 'C:\\ProgramData\\MySQL\\MySQL Server 8.0\\Uploads\\municipality_table.csv'
INTO TABLE municipality
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS
(province_id, municipality_id, municipality_name);

LOAD DATA LOCAL INFILE 'C:\\ProgramData\\MySQL\\MySQL Server 8.0\\Uploads\\barangay_table.csv'
INTO TABLE barangay
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS
(barangay_id, barangay_name, municipality_id);

SELECT * FROM ph_provinces.barangay AS b
INNER JOIN ph_provinces.municipality AS m 
ON b.municipality_id = m.municipality_id
INNER JOIN ph_provinces.province AS p
ON m.province_id = p.province_id
INNER JOIN ph_provinces.region AS r
ON p.region_id = r.region_id;