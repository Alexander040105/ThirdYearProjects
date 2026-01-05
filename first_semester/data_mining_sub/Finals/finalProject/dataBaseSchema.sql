CREATE DATABASE finals_data_mining;
CREATE TABLE finals_data_mining.students_dataset (
    Gender VARCHAR(10),
    Age INT,
    Department VARCHAR(100),
    Attendance_Percentage DECIMAL(5,2),
    Midterm_Score DECIMAL(5,2),
    Final_Score DECIMAL(5,2),
    Assignments_Avg DECIMAL(5,2),
    Quizzes_Avg DECIMAL(5,2),
    Participation_Score DECIMAL(5,2),
    Projects_Score DECIMAL(5,2),
    Total_Score DECIMAL(6,2),
    Grade VARCHAR(5),
    Study_Hours_per_Week DECIMAL(5,2),
    Extracurricular_Activities VARCHAR(100),
    Internet_Access_at_Home VARCHAR(50),
    Parent_Education_Level VARCHAR(100),
    Family_Income_Level VARCHAR(100),
    Stress_Level INT,
    Sleep_Hours_per_Night DECIMAL(4,2)
);

SELECT * FROM finals_data_mining.students_dataset;