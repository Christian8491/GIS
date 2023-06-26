# Installing PostgreSQL

In this tutorial we are going to install [**PostgreSQL**](https://www.postgresql.org/) an open source object-relational database system, only for Ubuntu. For Windows we can find out many resources on the internet about how to get PostgreSQL.

1. First, let's update our Ubuntu system to get newest versions of packages:
```
$ sudo apt update
```

2. To install PostgreSQL on Ubuntu we need to run via command line:
```
$ sudo apt install postgresql postgresql-contrib
```

3. Now we should be able to see the PostgreSQL version using:
```
$ psql --version
```

4. To connect to PostgreSQL via the command line
```
$ sudo -u postgres psql
```

Then will be able to see the Postgres interface
```
postgres=#
```

5. Unlike Windows we should set up our password literally, let's do it using the next statement (by default, our *postgres_user* will be **postgres**):
```
postgres=# ALTER USER <postgres_user> PASSWORD '<password>';
```


We can check the PostgreSQL version using SQL statement also:
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

To exit from the Postres interface we can use:
```
postgres=# \q
```
