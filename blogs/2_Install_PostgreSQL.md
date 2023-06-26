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
1. To list all the databases:
```
postgres=# \l
```

2. To display information about the connection (like the user and port):
```
postgres=#\conninfo
```

3. If we want to connect to one of these databases (select one of the listed databases, for instance **sdb**):
```
postgres=# \c sbd
```

4. To list all tables of **sbd** database
```
sdb=# \dt
              List of relations
 Schema |      Name       | Type  |  Owner   
--------+-----------------+-------+----------
 public | spatial_ref_sys | table | postgres
```

5. Finally we can execute some *SQL* statements:
```
sdb=# select srid, auth_name, auth_srid from spatial_ref_sys limit 5;
 srid | auth_name | auth_srid 
------+-----------+-----------
 2000 | EPSG      |      2000
 2001 | EPSG      |      2001
 2002 | EPSG      |      2002
 2003 | EPSG      |      2003
 2004 | EPSG      |      2004
(5 rows)
```
