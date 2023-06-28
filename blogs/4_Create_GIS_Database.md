# Creating a GIS Database

In this tutorial we are going to feed a spatial database with data where one of the columns hold a spatial type.

In our previous tutorial we created a GIS database (**sdb**). Using the same database let's create some tables with different geometries like Point, LineString and Polygon. For those who are not familiar with these geometric terms can visit the [PostGIs' Data Management](https://postgis.net/docs/using_postgis_dbmanagement.html).

We are removing the ```sdb=# ``` prefix from now because we are assuming some are using pgAdming platform.

### Point Geometry
Creating a table which contains a **Point** column:
```
CREATE TABLE point_table (city VARCHAR, country VARCHAR, geom geometry);
```

The default spatial projection is 4326, asuming we want to use another one we can manually tell PostGIS which one:
```
CREATE TABLE point_table (city VARCHAR, country VARCHAR, geom geometry(Point, 26918));
```

But for now lets avoid using a particular SRID and we will focus only on the *SQL* part.

Now, let's feed some values into our **point_table** table with few important cities:
```
INSERT INTO point_table VALUES
('Bogota', 'COL', 'POINT(4.59642 -74.08334)'),
('Lima', 'PER', 'POINT(-12.04801 -77.05006)'),
('Las Vegas', 'USA', 'POINT(36.21 -115.22001)'),
('Santiago', 'CHL', 'POINT(-33.45001 -70.66704)'),
('Rio de Janeiro', 'BRA', 'POINT(-22.92502 -43.22502)');
```

To check out whether our rows were added correctly, we can use:
```
SELECT city, country, ST_AsText(geom) FROM point_table;
```

That will show:
```
Bogota         | COL     | POINT(4.59642 -74.08334)
Lima           | PER     | POINT(-12.04801 -77.05006)
Las Vegas      | USA     | POINT(36.21 -115.22001)
Santiago       | CHL     | POINT(-33.45001 -70.66704)
Rio de Janeiro | BRA     | POINT(-22.92502 -43.22502)
 ```

The *ST_AsText()* method returns a well-known text (WKT) representation of the current geometry.

Now, let's check information like the type of geometry, X coordinate and Y coordinate.
```
SELECT ST_GeometryType(geom), ST_X(geom), ST_Y(geom) FROM point_table;
```

The result for this query is:
```
ST_Point        |   4.59642 |  -74.08334
ST_Point        | -12.04801 |  -77.05006
ST_Point        |     36.21 | -115.22001
ST_Point        | -33.45001 |  -70.66704
ST_Point        | -22.92502 |  -43.22502
```


### LineString Geometry
We can also create a table containing a **LineString** column:
```
CREATE TABLE line_table (city VARCHAR, country VARCHAR, geom geometry);
```


### Polygon Geometry




\dt to see all the tables


### Multi Geometry
Let's put all columns together



