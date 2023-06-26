# Installing PostgreSQL

In this tutorial we are going to install [**PostgreSQL**](https://www.postgresql.org/), an open source object-relational database system, only for Ubuntu. For Windows we can find out many resources on the internet about how to get PostgreSQL.

1. First, let's update our Ubuntu system to get newest versions of packages:
```
$ sudo apt update
```

2. To install PostgreSQL we need to run via command line:
```
$ sudo apt install postgresql postgresql-contrib
```

3. Now, we should be able to see the Postgres version using:
```
$ psql --version
```

4. To connect to Postgres via the command line:
```
$ sudo -u postgres psql
```

Then we will be able to see the Postgres interface:
```
postgres=#
```

5. Unlike Windows we should set up our password manually, let's do it using the next statement (by default, *postgres_user* will be **postgres**):
```
postgres=# ALTER USER <postgres_user> PASSWORD '<password>';
```

We can check the Postgres version using *SQL* statement also:
```
postgres=# select version();
```

Then we are going to get something like this:

```
                                                                version                                                                 
----------------------------------------------------------------------------------------------------------------------------------------
 PostgreSQL 14.8 (Ubuntu 14.8-0ubuntu0.22.04.1) on x86_64-pc-linux-gnu, compiled by gcc (Ubuntu 11.3.0-1ubuntu1~22.04.1) 11.3.0, 64-bit
(1 row)
```

To exit from the Postgres interface we can use:
```
postgres=# \q
```

## Useful commands
1. To display information about the connection (like the user and port):
```
postgres=# \conninfo
```

2. We can create a database called **sdb** using *SQL*:
```
postgres=# CREATE DATABASE sdb WITH OWNER = postgres;
```

3. To list all the databases:
```
postgres=# \l
```

4. To connect to one of these databases (select one from the above list, for instance **sdb**):
```
postgres=# \c sdb
```

5. Let's create a table **sdb_table** and a couple of columns to our **sdb** database:
```
sdb=# CREATE TABLE sdb_table (id VARCHAR, latitude DECIMAL(6,2));
```

6. To list all **sdb**'s tables:
```
sdb=# \dt
           List of relations
 Schema |   Name    | Type  |  Owner   
--------+-----------+-------+----------
 public | sdb_table | table | postgres
(1 row)
```

7. We can execute some *SQL* statements now:
```
sdb=# select * from sdb_table;
 id | latitude 
----+----------
(0 rows)
```

8. To remove our recent created table **sdb_table**: 
```
sdb=# drop table sdb_table;
```

9. This time we woun't find tables related:
```
sdb=# \dt
```

10. To remove our recent created database **sdb**, we will need first to switch to another database:
```
sdb=# \c postgres
postgres=# drop database sdb;
```

11. Finaly we can check whether the **sdb** database was removed:
```
postgres=# \l
```
