# Creating a GIS Database

In this tutorial we are going to feed a spatial database with data where one of the columns hold a spatial type.

In our previous tutorial we created a GIS database (**sdb**). Using the same database let's create some tables with different geometries like **Point**, **LineString** and **Polygon**. For those who are not familiar with these geometric terms can visit the [PostGIs' Data Management](https://postgis.net/docs/using_postgis_dbmanagement.html).

We are removing the ```sdb=#``` prefix from now because we are assuming some people are using the **pgAdming** platform.


## Point Geometry
Creating a table which contains a **Point** column:
```
CREATE TABLE point_table (city VARCHAR, country VARCHAR, geom geometry);
```

The default spatial projection is 4326, asuming we want to use another one we can manually select one:
```
CREATE TABLE point_table (city VARCHAR, country VARCHAR, geom GEOMETRY(POINT, 26918));
```

But for now let's avoid using a particular SRID and we will focus only on the *SQL* part.

Now, let's feed some values into our **point_table** table:
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

Now, let's check out information like the type of geometry, X coordinate and Y coordinate.
```
SELECT ST_GeometryType(geom), ST_X(geom), ST_Y(geom) FROM point_table;
```

The result for the query:
```
ST_Point        |   4.59642 |  -74.08334
ST_Point        | -12.04801 |  -77.05006
ST_Point        |     36.21 | -115.22001
ST_Point        | -33.45001 |  -70.66704
ST_Point        | -22.92502 |  -43.22502
```


## LineString Geometry
We can also create a table containing a **LineString** column:
```
CREATE TABLE line_table (segment VARCHAR, geom geometry);
```

Again, we can specify a spatial projection:
```
CREATE TABLE line_table (segment VARCHAR, geom GEOMETRY(LINESTRING, 4326));
```

Let's feed some values into our **line_table** table according to the below image:

![LineString](/blogs/imgs/linestring.png)

```
INSERT INTO line_table VALUES
('AB', 'LINESTRING(0 0, 1 1)'),
('BC', 'LINESTRING(1 1, 3 4)'),
('CD', 'LINESTRING(3 4, 7 4)'),
('DE', 'LINESTRING(7 4, 9 2)');
```

Checking out these values:
```
SELECT segment, ST_AsText(geom) FROM line_table;
```

Will display:
```
AB      | LINESTRING(0 0,1 1)
BC      | LINESTRING(1 1,3 4)
CD      | LINESTRING(3 4,7 4)
DE      | LINESTRING(7 4,9 2)
```

Finally, we can check out useful information like the type of geometry, the lenght of the lines, the first point from each line and the number of points these lines contain.
```
SELECT ST_GeometryType(geom), ST_Length(geom), ST_AsText(ST_StartPoint(geom)), ST_NPoints(geom) FROM line_table;
```

The result for the query:
```
ST_LineString   | 1.4142135623730951 | POINT(0 0) |          2
ST_LineString   |  3.605551275463989 | POINT(1 1) |          2
ST_LineString   |                  4 | POINT(3 4) |          2
ST_LineString   | 2.8284271247461903 | POINT(7 4) |          2
```


## Polygon Geometry
We can use any of these to create a table holding a **Polygon**:
```
CREATE TABLE polygon_table (name VARCHAR, geom geometry);
```
```
CREATE TABLE polygon_table (name VARCHAR, geom GEOMETRY(POLYGON, 4326));
```

Feeding some values into our **polygon_table** table according to the below image:
![LineString](/blogs/imgs/polygon.png)

```
INSERT INTO polygon_table VALUES
('Polygon1', 'POLYGON((0 0, 1 1, 3 4, 7 4, 9 2, 9 0, 0 0))');
```

Checking out these values:
```
SELECT name, ST_AsText(geom) FROM polygon_table;
```

Will display:
```
Polygon1 | POLYGON((0 0,1 1,3 4,7 4,9 2,9 0,0 0))
```

As before, let's get interesting information like the type of geometry, the perimeter and the area as well.
```
SELECT ST_GeometryType(geom), ST_Perimeter(geom), ST_Area(geom) FROM polygon_table;
```

The result is:
```
ST_Polygon      | 22.84819196258327 |    27.5
```


### Multi Geometry
\dt to see all the tables
Note that the column **geom** can be used for different type of geometry, so let's create multiple geometries.
Let's put all columns together

