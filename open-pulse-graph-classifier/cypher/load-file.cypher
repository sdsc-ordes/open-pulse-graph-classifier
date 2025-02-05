LOAD CSV WITH HEADERS FROM 'file:///example_data.csv' AS row

// Create Source Node
MERGE (s:Node {name: row.Source})
SET s:Person  WHERE row.Source_type = 'Person'
SET s:Organisation WHERE row.Source_type = 'Organisation'
SET s:Repository WHERE row.Source_type = 'Repo'

// Create Target Node
MERGE (t:Node {name: row.Target})
SET t:Person  WHERE row.Target_type = 'Person'
SET t:Organisation WHERE row.Target_type = 'Organisation'
SET t:Repository WHERE row.Target_type = 'Repo'

// Create Bidirectional Relationship
MERGE (s)-[r1:CONNECTED_TO]->(t)
SET r1.property_relationship = row.Property_relationship

MERGE (t)-[r2:CONNECTED_TO]->(s)
SET r2.property_relationship = row.Property_relationship;
