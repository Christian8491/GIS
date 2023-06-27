# Enabling PostGIS

In this tutorial we are going to install [**PostGIS**](http://postgis.net), which extends PostgreSQL by adding geospatial support.

1. Installing PostGIS should be as easy as (make sure we already installed PostgreSQL):
```
$ sudo apt install postgis
```

## Useful commands

1. Let's create a database as we did in our [previous blog] (https://github.com/Christian8491/GIS/blob/master/blogs/2_Install_PostgreSQL.md). 
```
postgres=# CREATE DATABASE sdb WITH OWNER = postgres;
```

2. By default, no database has the PostGIS extension enabled, we can enable it (connecting previously to our **sdb** database):
```
postgres=# \c sdb
sdb=# CREATE EXTENSION postgis;
```

Once we create the extension, we can check that a default **spatial_ref_sys** table was created.

3. To list all tables of **sdb** database:
```
sdb=# \dt
              List of relations
 Schema |      Name       | Type  |  Owner   
--------+-----------------+-------+----------
 public | spatial_ref_sys | table | postgres
(1 row)
```

4. Now we can execute some *SQL* statements:
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

We can then remove the table as we did in our previous blog.



### Installing pgAdmin

We can interact directly on our PostgreSQL's command line, but we can also use [pgAdmin] (https://www.pgadmin.org/download/pgadmin-4-apt/) (read the instructions to get it). Using pgAdmin is friendly and more intuitive.
