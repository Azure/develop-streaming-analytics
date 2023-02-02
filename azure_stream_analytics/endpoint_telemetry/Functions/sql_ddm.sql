
    CREATE TABLE endpoint_conf_change (
    ep_id VARCHAR(50) NOT NULL,
    eventtime DATETIME NOT NULL,
    attr_name VARCHAR(50) NULL,
    attr_value VARCHAR(100) NULL,

);



  CREATE TABLE endpoint_configuration (
    ep_id VARCHAR(50) NOT NULL,
    eventtime DATETIME NOT NULL,
    attr_1 VARCHAR(50) NULL,
    attr_2 VARCHAR(50) NULL,
    attr_3 VARCHAR(50) NULL,
    attr_4 VARCHAR(50) NULL,
    attr_5 VARCHAR(50) NULL,
    attr_6 VARCHAR(50) NULL,
    attr_7 VARCHAR(50) NULL,
    attr_8 VARCHAR(50) NULL,
    attr_9 VARCHAR(50) NULL,
    attr_10 VARCHAR(50) NULL
);


CREATE TRIGGER tr_endpoint_configuration_update ON endpoint_conf_change
AFTER INSERT
AS
BEGIN
DECLARE @ep_id VARCHAR(50), @eventtime DATETIME, @attr_name VARCHAR(50), @attr_value VARCHAR(100)


SELECT @ep_id = ep_id, @eventtime = eventtime, @attr_name = attr_name, @attr_value = attr_value
FROM inserted

DECLARE @column_name VARCHAR(50)

SET @column_name = 'attr_' + RIGHT(@attr_name, CHARINDEX('_', REVERSE(@attr_name)) - 1)

DECLARE @last_record endpoint_configuration

SELECT TOP 1 @last_record.*
FROM endpoint_configuration
WHERE ep_id = @ep_id
ORDER BY eventtime DESC

DECLARE @new_record endpoint_configuration

SET @new_record = @last_record
SET @new_record.eventtime = @eventtime
SET @new_record.@column_name = @attr_value

INSERT INTO endpoint_configuration
SELECT @new_record.*



