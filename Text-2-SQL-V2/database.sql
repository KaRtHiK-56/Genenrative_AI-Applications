-- Create the employee table
CREATE TABLE employee (
    emp_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    age INTEGER,
    gender TEXT,
    salary REAL
);

-- Insert 20 records into the employee table
INSERT INTO employee (emp_id, name, age, gender, salary) VALUES
(1, 'Alice', 30, 'Female', 60000.00),
(2, 'Bob', 34, 'Male', 70000.00),
(3, 'Charlie', 28, 'Male', 50000.00),
(4, 'Diana', 45, 'Female', 80000.00),
(5, 'Eve', 29, 'Female', 55000.00),
(6, 'Frank', 33, 'Male', 72000.00),
(7, 'Grace', 40, 'Female', 68000.00),
(8, 'Henry', 50, 'Male', 90000.00),
(9, 'Isabella', 27, 'Female', 53000.00),
(10, 'Jack', 35, 'Male', 75000.00),
(11, 'Kate', 42, 'Female', 82000.00),
(12, 'Leo', 31, 'Male', 64000.00),
(13, 'Mona', 26, 'Female', 51000.00),
(14, 'Nate', 38, 'Male', 78000.00),
(15, 'Olivia', 32, 'Female', 61000.00),
(16, 'Paul', 47, 'Male', 85000.00),
(17, 'Quinn', 28, 'Female', 57000.00),
(18, 'Ray', 29, 'Male', 56000.00),
(19, 'Sophie', 36, 'Female', 73000.00),
(20, 'Tom', 44, 'Male', 83000.00);
