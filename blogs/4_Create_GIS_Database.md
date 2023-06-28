# Creating a GIS Database

In this tutorial we are going to feed a spatial database with fake data where one of the columns hold a spatial type.

In our previous tutorial we created a GIS database (**sdb**). Using the same database let's create some tables with differente geometries like Point, LineString and Polygon. For those who are not familiar with the previous geometric terms can visit the [PostGIs' Data Management](https://postgis.net/docs/using_postgis_dbmanagement.html).


### Point Geometry
We can create a table which contains a **Point** column using:
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

The *ST_AsText()* method returns a well-known text (WKT) representation of the current geometry.

Now, let's check information like the kind of geometry, X coordinate and Y coordinate.
```
SELECT ST_GeometryType(geom), ST_X(geom) AS latitude, ST_Y(geom) AS longitude FROM point_table;
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



