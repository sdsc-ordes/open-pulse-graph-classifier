LOAD CSV WITH HEADERS FROM 'https://raw.githubusercontent.com/CayleyCausey/ScoobyGraph/main/scoobydoo_new.csv' as row
MATCH (s:Series {seriesName:row.series_name})
MERGE (e:Episode {episodeTitle:row.title})
SET e.season = row.season, e.network = row.network,
e.imdb = row.imdb, e.engagement = row.engagement,
e.runtime = row.runtime, e.format = row.format
MERGE (s) <-[:IN_SERIES] - (e)
