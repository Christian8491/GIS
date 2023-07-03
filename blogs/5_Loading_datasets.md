# Loading a dataset

In this tutorial we are going to import a dataset which contains latitudes and longitudes from a CSV file. After that we are going to transform our coordinates into a new column consisting of a **geometry** type.

We are using the [cities.csv](/blogs/datasets/cities.csv) dataset for this example which holds 6 columns: {*id, name, country, latitude, longitude, population*}.

Our first step is to create a table (once we moved over the **sdb** database) holding the same fields that the csv file has.
```
CREATE TABLE cities (id INTEGER, name VARCHAR, country VARCHAR, latitude REAL, longitude REAL, population BIGINT, coordinate geometry);
```

For more details about [PostgreSQL numeric datatypes](https://www.postgresql.org/docs/current/datatype-numeric.html). Note that the **cities.csv** dataset doesn't have any geometry column, so it's expected that the **coordinate** column of the **cities** table doesn't hold any data yet.

To check the current data types of each column:
```
sdb=# \d cities
```

Then, it will show us something like:
```
                      Table "public.cities"
   Column   |       Type        | Collation | Nullable | Default 
------------+-------------------+-----------+----------+---------
 id         | integer           |           |          | 
 name       | character varying |           |          | 
 country    | character varying |           |          | 
 latitude   | real              |           |          | 
 longitude  | real              |           |          | 
 population | bigint            |           |          | 
 coordinate | geometry          |           |          | 
```

The second step is to copy our local dataset to PostgreSQL:

```
\copy cities(id,name,country,latitude,longitude,population) FROM '/local/path/to/cities.csv' DELIMITERS ',' CSV HEADER;
```

The DELIMITERS ',' will let PostgreSQL know the kind of delimiter, the **CSV** tells what type of file is and the **HEADER** argument lets PostreSQL that the file contains the header as the first line. If run into troubles like **permission denied** try to move the **cities.csv** file to the **/tmp/** directory because this is accessible by all users.

Now, let's see how our **cities** table looks:
```
SELECT * FROM cities LIMIT 5;
```

We should get:
```
 id |    name     | country | latitude | longitude | population | coordinate 
----+-------------+---------+----------+-----------+------------+------------
  1 | Bombo       | UGA     |   0.5833 |   32.5333 |      75000 | 
  2 | Fort Portal | UGA     |    0.671 |    30.275 |      42670 | 
  3 | Potenza     | ITA     |   40.642 |    15.799 |      69060 | 
  4 | Campobasso  | ITA     |   41.563 |    14.656 |      50762 | 
  5 | Aosta       | ITA     |   45.737 |     7.315 |      34062 | 
(5 rows)
```

As we can observe, the **coordinate** column is empty for now.

Let's check the total number of registers:
```
sdb=# SELECT count(*) from cities;
 count 
-------
  1249
(1 row)
```

The next step is fill out the **coordinate** column using both **longitude** and **latitude** columns:

```
UPDATE cities SET coordinate = ST_GeomFromText('POINT(' || latitude || ' ' || longitude || ')', 4326);
```

We can check again our results:
```
SELECT id, name, country, latitude, longitude, population, ST_AsText(coordinate) FROM cities LIMIT 5;
```

This time we should get:
```
 id |    name     | country | latitude | longitude | population |       st_astext       
----+-------------+---------+----------+-----------+------------+-----------------------
  1 | Bombo       | UGA     |   0.5833 |   32.5333 |      75000 | POINT(0.5833 32.5333)
  2 | Fort Portal | UGA     |    0.671 |    30.275 |      42670 | POINT(0.671 30.275)
  3 | Potenza     | ITA     |   40.642 |    15.799 |      69060 | POINT(40.642 15.799)
  4 | Campobasso  | ITA     |   41.563 |    14.656 |      50762 | POINT(41.563 14.656)
  5 | Aosta       | ITA     |   45.737 |     7.315 |      34062 | POINT(45.737 7.315)
(5 rows)
```