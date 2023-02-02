
WITH cte AS (
    SELECT ep_id, MAX(eventtime) as max_eventtime
    FROM endpoint_configuration
    GROUP BY ep_id
)
SELECT ec.*
FROM endpoint_configuration ec
JOIN cte ON ec.ep_id = cte.ep_id AND ec.eventtime = cte.max_eventtime;

