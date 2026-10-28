-- Build the modelling table from the raw freMTPL2 frequency + severity tables.
-- Cleaning rules (standard for freMTPL2):
--   * ClaimNb capped at 4, Exposure capped at 1 policy-year, drop non-positive exposure
--   * VehGas quote characters removed
--   * claim amounts capped at 100,000 each and zeroed when the policy reports no claim
DROP TABLE IF EXISTS policies;
CREATE TABLE policies AS
SELECT
    f.IDpol,
    MIN(f.ClaimNb, 4)                         AS ClaimNb,
    MIN(f.Exposure, 1.0)                      AS Exposure,
    f.Area,
    CAST(f.VehPower AS INTEGER)               AS VehPower,
    CAST(f.VehAge AS INTEGER)                 AS VehAge,
    CAST(f.DrivAge AS INTEGER)                AS DrivAge,
    CAST(f.BonusMalus AS INTEGER)             AS BonusMalus,
    f.VehBrand,
    REPLACE(f.VehGas, '''', '')               AS VehGas,
    CAST(f.Density AS INTEGER)                AS Density,
    f.Region,
    CASE WHEN f.ClaimNb > 0 THEN COALESCE(s.amt, 0) ELSE 0 END AS ClaimAmount,
    CASE WHEN f.ClaimNb > 0 THEN COALESCE(s.n, 0)   ELSE 0 END AS NSev
FROM freq_raw f
LEFT JOIN (
    SELECT IDpol, SUM(MIN(ClaimAmount, 100000)) AS amt, COUNT(*) AS n
    FROM sev_raw
    GROUP BY IDpol
) s ON s.IDpol = f.IDpol
WHERE f.Exposure > 0;
