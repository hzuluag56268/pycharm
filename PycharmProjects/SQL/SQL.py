SELECT title, release_year, country
FROM films;


SELECT *
FROM films;




SELECT DISTINCT certification
FROM films;    #unique in pd



For example, this code gives the number of rows in the people table:

SELECT COUNT(*)
FROM people;




count the number of non - missing values in a particular column,

SELECT COUNT(birthdate)
FROM people;




For example, this query counts the number of
distinct birth dates contained in the people table:

SELECT COUNT(DISTINCT birthdate)
FROM people;




'''''''''Filtering results
'''

SELECT title
FROM films
WHERE title = 'Metropolis';


SELECT COUNT(*)
FROM films
WHERE release_year < 2000;


SELECT title, release_year
FROM films
WHERE release_year < 2000
AND language = 'Spanish';







WHERE release_year = 1994
OR release_year = 2000;


WHERE (release_year >= 1990 AND release_year < 2000)
AND (language = 'French' OR language = 'Spanish')
AND gross > 2000000;




WHERE release_year BETWEEN 1990 AND 2000


WHERE language IN ('English', 'Spanish', 'French');   #.asin() in pd





SELECT name
FROM people
WHERE deathdate IS NULL;

SELECT name
FROM people
WHERE birthdate IS NOT NULL;




 the following query matches companies like
'Data', 'DataC' 'DataCamp', 'DataMind', and so on:


WHERE name LIKE 'Data%';


The _ wildcard will match a single character.
For example, the following query matches companies
like 'DataCamp', 'DataComp', and so on:


WHERE name LIKE 'DataC_mp';
You can also use the NOT LIKE operator


WHERE name LIKE '_r%';





'''3 Aggregate Functions
'''




..........Aggregate Functions


SELECT AVG(budget)
FROM films;

SUM()
MAX()
MIN()



SELECT AVG(duration) / 60.0 AS avg_duration_hours  #alias
FROM films


SELECT COUNT(deathdate) * 100.0 / COUNT(*)  AS percentage_dead
FROM people




'''Sorting and grouping
'''



......Sorting and grouping

SELECT *
FROM films
WHERE release_year <> 2015
ORDER BY duration;


SELECT title, duration
FROM films
ORDER BY duration DESC;


SELECT certification, release_year, title
FROM films
ORDER BY certification, release_year;



SELECT imdb_score, COUNT(*)
FROM reviews
GROUP BY imdb_score;


SELECT country, release_year, MIN(gross)
FROM films
GROUP BY country, release_year
ORDER BY country, release_year;





SELECT release_year
FROM films
GROUP BY release_year
HAVING COUNT(title) > 10;


SELECT release_year, AVG(budget) as avg_budget, AVG(gross) as avg_gross
FROM films
where release_year > 1990
GROUP BY release_year
HAVING AVG(budget) > 60000000      #having modify after grouping
ORDER BY AVG(gross) DESC


-- select country, average budget, average gross
SELECT country, AVG(budget) AS avg_budget, AVG(gross) AS avg_gross
-- from the films table
FROM films
-- group by country
GROUP BY country
-- where the country has more than 10 titles
HAVING COUNT(country) > 10
-- order by country
ORDER BY country
-- limit to only show 5 results
LIMIT 5