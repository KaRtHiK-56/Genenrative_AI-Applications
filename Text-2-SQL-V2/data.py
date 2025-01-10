import sqlite3 
  
# Connecting to sqlite 
conn = sqlite3.connect('database.db') 
  
# Creating a cursor object using the  
# cursor() method 
cursor = conn.cursor() 
  
# Creating table 
table ="""CREATE TABLE EMPLOYEE(
emp_name char(20) Not null,
emp_id int,
salary float default(50000),
bonus char(10),
dept_id int
);"""
cursor.execute(table) 
 
#Queries to INSERT records. 
cursor.execute('''insert into EMPLOYEE values("Harvey Specter",'1111111111','50000','not null','101');''')
cursor.execute(''' insert into EMPLOYEE values("Monica Geller",'1111111112','75000','null','102'); ''')
cursor.execute(''' insert into EMPLOYEE values("Ross Geller",'1111111113','60000','not null','103'); ''')
cursor.execute(''' insert into EMPLOYEE values("Rachel Green",'1111111114','50000','null','104'); ''')
cursor.execute(''' insert into EMPLOYEE values("Phoebe Buffay",'1111111115','50000','not null','101'); ''')
cursor.execute(''' insert into EMPLOYEE values("Joey Tribbiani",'1111111116','100000','null','102'); ''')
cursor.execute(''' insert into EMPLOYEE values("Mike Ross",'1111111117','50000','not null','103'); ''')
cursor.execute(''' insert into EMPLOYEE values("Rachel Zane",'1111111118','75000','null','104'); ''')
cursor.execute(''' insert into EMPLOYEE values("Monica Bing",'1111111119','20000','not null','101'); ''')
cursor.execute(''' insert into EMPLOYEE values("Chandler Bing",'1111111110','30000','null','102'); ''')
# Display data inserted 
print("Data Inserted in the table: ") 
data=cursor.execute('SELECT * FROM products') 
for row in data: 
    print(row) 
# Commit your changes in the database     
conn.commit() 
  
# Closing the connection 
conn.close()



  
