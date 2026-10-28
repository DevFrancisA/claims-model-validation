# Optional R cross-check of the SQL cleaning step (not required to run the app).
# Rscript r/clean_check.R  -> compares policy / claim totals against the SQLite table.
library(DBI); library(RSQLite)
con <- dbConnect(SQLite(), "data/claims.db")
p <- dbGetQuery(con, "SELECT * FROM policies")
f <- read.csv("data/freMTPL2freq.csv")
stopifnot(nrow(p) == sum(f$Exposure > 0))
cat(sprintf("policies=%d claims=%d exposure=%.0f\n", nrow(p), sum(p$ClaimNb), sum(p$Exposure)))
print(aggregate(cbind(ClaimNb, Exposure) ~ cut(DrivAge, c(17,25,35,50,70,100)), p, sum))
dbDisconnect(con)
