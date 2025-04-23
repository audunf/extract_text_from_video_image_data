



# Analyzing timestamp output with DuckDB

The output in ```results.csv``` might look like the following for video files where timestamps are recognized.

```
"Video Time","Frame","Filename","ROI ID","Recognized Text"
"00:00:00.039","1","202547582.avi","0","13:40:30.43"
"00:00:00.039","1","2025947582.avi","1","2025-04-10 15:40:30.29"
"00:00:00.039","1","2025947582.avi","2","2025-04-10 15:40:30.30"
"00:00:00.039","1","2025947582.avi","3","2025-04-10 15:40:30.28"
"00:00:00.039","1","2025947582.avi","4","2025-04-10 15:40:30.30"
"00:00:10.033","251","2025947582.avi","0","13:40:40.43"
"00:00:10.033","251","2025947582.avi","1","2025-04-10 15:40:40.28"
"00:00:10.033","251","2025947582.avi","2","2025-04-10 15:40:40.30"
"00:00:10.033","251","2025947582.avi","3","2025-04-10 15:40:40.27"
"00:00:10.033","251","2025947582.avi","4","2025-04-10 15:40:40.30"
```
Locating difference above 1 second between timestamps within a single frame in the same file.



```sql
.maxrows -1

-- Step 1: Read data and extract the mm:ss.ff part
WITH RawData AS (
    SELECT * FROM read_csv_auto('results.csv', header=true)
),
ExtractedTimes AS (
    SELECT
        "Filename",
        "Frame",
        "ROI ID",
        "Recognized Text", -- Keep original text
        -- Extract minutes, seconds, and centiseconds (ff) using capture groups
        -- Assumes the pattern is always at the very end of the string
        regexp_extract("Recognized Text", '.*?(\d{1,2}):(\d{2})\.(\d{2})$', 1) AS extracted_minutes_str,
        regexp_extract("Recognized Text", '.*?(\d{1,2}):(\d{2})\.(\d{2})$', 2) AS extracted_seconds_str,
        regexp_extract("Recognized Text", '.*?(\d{1,2}):(\d{2})\.(\d{2})$', 3) AS extracted_centiseconds_str
    FROM RawData
),
-- Step 2: Parse extracted parts and calculate total milliseconds (ignoring hours/date)
TimeValues AS (
    SELECT
        "Filename",
        "Frame",
        "ROI ID",
        "Recognized Text", -- Pass original text through
        extracted_minutes_str,
        extracted_seconds_str,
        extracted_centiseconds_str,
        -- Try casting parts to integers
        try_cast(extracted_minutes_str AS INTEGER) AS parsed_minutes,
        try_cast(extracted_seconds_str AS INTEGER) AS parsed_seconds,
        try_cast(extracted_centiseconds_str AS INTEGER) AS parsed_centiseconds
    FROM ExtractedTimes
),
-- Step 3: Calculate comparable milliseconds and filter invalid parses
ComparableMilliseconds AS (
    SELECT
        "Filename",
        "Frame",
        "ROI ID",
        "Recognized Text", -- Pass original text through
        -- Calculate total milliseconds: (min * 60 * 1000) + (sec * 1000) + (centi * 10)
        (parsed_minutes * 60000 + parsed_seconds * 1000 + parsed_centiseconds * 10)::BIGINT AS comparable_total_ms
    FROM TimeValues
    -- Ensure all parts were parsed successfully
    WHERE parsed_minutes IS NOT NULL
      AND parsed_seconds IS NOT NULL
      AND parsed_centiseconds IS NOT NULL
),
-- Step 4: Calculate differences between ROI 0 and ROIs 1, 2, 3, 4 within the same Filename/Frame
Differences AS (
    SELECT
        t1."Filename",
        t1."Frame",
        t1."ROI ID" AS ROI1, -- This will always be 0
        t1."Recognized Text" AS Text1, -- Get Text for ROI 0
        t1.comparable_total_ms AS Time1_ms,
        t2."ROI ID" AS ROI2, -- This will be 1, 2, 3, or 4
        t2."Recognized Text" AS Text2, -- Get Text for ROI 1/2/3/4
        t2.comparable_total_ms AS Time2_ms,
        -- Calculate the absolute difference in milliseconds
        abs(t1.comparable_total_ms - t2.comparable_total_ms) AS diff_total_ms
    FROM ComparableMilliseconds t1
    JOIN ComparableMilliseconds t2 ON t1."Filename" = t2."Filename" AND t1."Frame" = t2."Frame"
    -- Ensure t1 is always ROI 0 and t2 is one of the target ROIs (1, 2, 3, 4)
    WHERE t1."ROI ID" = 0 AND t2."ROI ID" IN (1, 2, 3, 4)
),
-- Step 5: Get the text for ROI ID 5 for relevant Filename/Frame combinations
ROI5Text AS (
    SELECT DISTINCT -- Use DISTINCT in case ROI 5 appears multiple times for the same frame
        "Filename",
        "Frame",
        "Recognized Text" AS Text5
    FROM ComparableMilliseconds
    WHERE "ROI ID" = 5
)
-- Step 6: Filter differences >= 200ms, join with ROI 5 text, and format the output
SELECT
    d."Filename",
    d."Frame",
    d.ROI1,
    d.Text1, -- Include original text for ROI1
    d.ROI2,
    d.Text2, -- Include original text for ROI2
    r5.Text5, -- Include original text for ROI5 (will be NULL if ROI5 doesn't exist for this frame)
    d.diff_total_ms, -- Show the raw difference in milliseconds
    -- Correctly calculate minute, second, and millisecond components from the total ms difference
    -- 1. Calculate total whole seconds
    floor(d.diff_total_ms / 1000)::BIGINT AS total_diff_seconds_int,
    -- 2. Extract minutes component (0-59) from total whole seconds
    floor(total_diff_seconds_int / 60)::BIGINT AS diff_minutes,
    -- 3. Extract seconds component (0-59) from total whole seconds
    total_diff_seconds_int % 60 AS diff_seconds,
    -- 4. Extract milliseconds (0-999) - Cast to BIGINT before formatting
    (d.diff_total_ms % 1000)::BIGINT AS diff_milliseconds,
    -- Format the difference string as requested
    format('{:d}min, {:d}sec, {:d}ms',
           diff_minutes,
           diff_seconds,
           diff_milliseconds -- This value is now guaranteed to be BIGINT
    ) AS formatted_difference
FROM Differences d -- Alias the Differences CTE
-- LEFT JOIN to include rows even if ROI 5 is missing for that frame
LEFT JOIN ROI5Text r5 ON d."Filename" = r5."Filename" AND d."Frame" = r5."Frame"
-- Filter condition: difference must be greater than or equal to 200 milliseconds
WHERE d.diff_total_ms >= 200
-- Order results for readability
ORDER BY
    d."Filename" DESC,
    d."Frame" ASC,
    d.diff_total_ms DESC,
    d.ROI1, -- ROI1 will always be 0 here
    d.ROI2;

```

Late:
│ sq48252.avi │  5241 │     0 │ 12:23:45.86 │     1 │ 14:23:45.62 │ FRONT_CAM_C22025-04-16 14:23:45.80  │           240 │                      0 │            0 │            0 │               240 │ 0min, 0sec, 240ms    │
│ sq48252.avi │  5241 │     0 │ 12:23:45.86 │     2 │ 14:23:45.62 │ FRONT_CAM_C22025-04-16 14:23:45.80  │           240 │                      0 │            0 │            0 │               240 │ 0min, 0sec, 240ms    │
│ sq48252.avi │  5241 │     0 │ 12:23:45.86 │     4 │ 14:23:45.65 │ FRONT_CAM_C22025-04-16 14:23:45.80  │           210 │                      0 │            0 │            0 │               210 │ 0min, 0sec, 210ms    │

│ sq48456.avi │  6996 │     0 │ 03:37:37.99 │     1 │ 05:37:37.73 │ FRONT_CAM_C22025-04-22 05:37:37.82  │           260 │                      0 │            0 │            0 │               260 │ 0min, 0sec, 260ms    │
│ sq48456.avi │  6996 │     0 │ 03:37:37.99 │     2 │ 05:37:37.76 │ FRONT_CAM_C22025-04-22 05:37:37.82  │           230 │                      0 │            0 │            0 │               230 │ 0min, 0sec, 230ms    │
│ sq48456.avi │  6996 │     0 │ 03:37:37.99 │     4 │ 05:37:37.77 │ FRONT_CAM_C22025-04-22 05:37:37.82  │           220 │                      0 │            0 │            0 │               220 │ 0min, 0sec, 220ms    │
│ sq48456.avi │  6996 │     0 │ 03:37:37.99 │     3 │ 05:37:37.79 │ FRONT_CAM_C22025-04-22 05:37:37.82  │           200 │                      0 │            0 │            0 │               200 │ 0min, 0sec, 200ms    │

│ sq48456.avi │  8748 │     0 │ 03:38:48.08 │     2 │ 05:38:47.82 │ FRONT_CAM_C22025-04-22 05:38:47.89  │           260 │                      0 │            0 │            0 │               260 │ 0min, 0sec, 260ms    │
│ sq48456.avi │  8748 │     0 │ 03:38:48.08 │     4 │ 05:38:47.83 │ FRONT_CAM_C22025-04-22 05:38:47.89  │           250 │                      0 │            0 │            0 │               250 │ 0min, 0sec, 250ms    │
│ sq48456.avi │  8748 │     0 │ 03:38:48.08 │     1 │ 05:38:47.84 │ FRONT_CAM_C22025-04-22 05:38:47.89  │           240 │                      0 │            0 │            0 │               240 │ 0min, 0sec, 240ms    │
│ sq48456.avi │  8748 │     0 │ 03:38:48.08 │     3 │ 05:38:47.86 │ FRONT_CAM_C22025-04-22 05:38:47.89  │           220 │                      0 │            0 │            0 │               220 │ 0min, 0sec, 220ms    │