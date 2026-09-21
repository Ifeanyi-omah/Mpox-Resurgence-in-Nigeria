###############################################################################
# MPXV Nigeria — dispersal statistics from continuous phylogeography
#
# Computes, per epidemic wave, from the posterior of an RRW analysis:
#   1. weighted lineage dispersal velocity   = sum(d) / sum(t)          [km/yr]
#   2. weighted diffusion coefficient        = sum(d^2) / (4 * sum(t))  [km^2/yr]
#   3. wavefront distance from epidemic origin through time             [km]
#
# Branches that straddle a wave boundary are CLIPPED at the boundary and the
# interpolated coordinate is used, so no branch contributes to two waves with
# its full length. Set CLIP_BRANCHES = FALSE to instead assign whole branches
# by their temporal midpoint (report both as a sensitivity check).
#
# NOTE ON COLUMN NAMES: seraphim's extraction files use columns along the lines
# of node1, node2, length, startLon, startLat, endLon, endLat, startYear,
# endYear. Run  head(read.csv(list.files(localTreesDirectory, full.names=TRUE)[1]))
# once and fix the COL_* constants below if they differ in your version.
###############################################################################

library(seraphim)
library(geosphere)
library(HDInterval)   # or use coda::HPDinterval

## ---------------------------------------------------------------- settings --
treeFile              <- "NCDC+CladeIIb+CONTphy+2025-04-04.trees"
localTreesDirectory   <- "Tree_extractions"
burnIn                <- 1001          # in TREES, not states — check your .log
nberOfTreesToSample   <- 1000
mostRecentSamplingDatum <- 2025.79     # <-- set to your true most recent tip date
coordinateAttributeName <- "location"  # trait name used in the BEAST XML
nberOfCores           <- 4

waveBreaks <- c(2016.0, 2021.0, 2024.0, 2026.0)   # Wave 1 / 2 / 3 boundaries
waveNames  <- c("Wave 1 (2016-2020)", "Wave 2 (2021-2023)", "Wave 3 (2024-2025)")
CLIP_BRANCHES <- TRUE

COL_SLON <- "startLon"; COL_SLAT <- "startLat"
COL_ELON <- "endLon";   COL_ELAT <- "endLat"
COL_ST   <- "startYear"; COL_ET  <- "endYear"

## ------------------------------------------------------- 1. tree extraction --
if (!dir.exists(localTreesDirectory)) {
  dir.create(localTreesDirectory)
  allTrees <- scan(file = treeFile, what = "", sep = "\n", quiet = TRUE)
  treeExtractions(localTreesDirectory, allTrees, burnIn,
                  randomSampling = FALSE, nberOfTreesToSample,
                  mostRecentSamplingDatum, coordinateAttributeName, nberOfCores)
}
extFiles <- list.files(localTreesDirectory, pattern = "\\.csv$", full.names = TRUE)
cat("Extraction files:", length(extFiles), "\n")

## ------------------------------------------- 2. per-branch distance and time --
geo_km <- function(lon1, lat1, lon2, lat2) {
  distVincentyEllipsoid(cbind(lon1, lat1), cbind(lon2, lat2)) / 1000
}

# Clip one branch to [t0, t1]; returns NULL if no overlap.
clip_branch <- function(sLon, sLat, eLon, eLat, sT, eT, t0, t1) {
  if (eT <= t0 || sT >= t1) return(NULL)
  dur <- eT - sT
  if (dur <= 0) return(NULL)
  a <- max(sT, t0); b <- min(eT, t1)
  fa <- (a - sT) / dur; fb <- (b - sT) / dur
  list(lon1 = sLon + fa * (eLon - sLon), lat1 = sLat + fa * (eLat - sLat),
       lon2 = sLon + fb * (eLon - sLon), lat2 = sLat + fb * (eLat - sLat),
       t = b - a)
}

wave_stats_one_tree <- function(tab, w) {
  t0 <- waveBreaks[w]; t1 <- waveBreaks[w + 1]
  sLon <- tab[[COL_SLON]]; sLat <- tab[[COL_SLAT]]
  eLon <- tab[[COL_ELON]]; eLat <- tab[[COL_ELAT]]
  sT   <- tab[[COL_ST]];   eT   <- tab[[COL_ET]]

  if (CLIP_BRANCHES) {
    d <- numeric(0); t <- numeric(0)
    for (i in seq_along(sT)) {
      cb <- clip_branch(sLon[i], sLat[i], eLon[i], eLat[i], sT[i], eT[i], t0, t1)
      if (is.null(cb) || cb$t <= 0) next
      d <- c(d, geo_km(cb$lon1, cb$lat1, cb$lon2, cb$lat2))
      t <- c(t, cb$t)
    }
  } else {
    mid  <- (sT + eT) / 2
    keep <- mid >= t0 & mid < t1 & (eT - sT) > 0
    d <- geo_km(sLon[keep], sLat[keep], eLon[keep], eLat[keep])
    t <- (eT - sT)[keep]
  }
  if (length(d) == 0) return(c(velocity = NA, diffusion = NA, nBranches = 0))
  c(velocity  = sum(d) / sum(t),           # km / yr
    diffusion = sum(d^2) / (4 * sum(t)),   # km^2 / yr
    nBranches = length(d))
}

## --------------------------------------------------- 3. loop over posterior --
res <- array(NA, dim = c(length(extFiles), length(waveNames), 3),
             dimnames = list(NULL, waveNames, c("velocity", "diffusion", "nBranches")))

for (j in seq_along(extFiles)) {
  tab <- read.csv(extFiles[j], header = TRUE)
  for (w in seq_along(waveNames)) res[j, w, ] <- wave_stats_one_tree(tab, w)
  if (j %% 100 == 0) cat("  tree", j, "\n")
}

summarise <- function(x) {
  x <- x[is.finite(x)]
  h <- hdi(x, credMass = 0.95)
  sprintf("%.1f (95%% HPD %.1f-%.1f)", median(x), h[1], h[2])
}

cat("\n=== Weighted lineage dispersal velocity (km/yr) ===\n")
for (w in seq_along(waveNames)) cat(waveNames[w], ":", summarise(res[, w, "velocity"]), "\n")

cat("\n=== Weighted diffusion coefficient (km^2/yr) ===\n")
for (w in seq_along(waveNames)) cat(waveNames[w], ":", summarise(res[, w, "diffusion"]), "\n")

cat("\n=== Posterior support for increases ===\n")
cat("P(v_W2 > v_W1) =", mean(res[, 2, "velocity"] > res[, 1, "velocity"], na.rm = TRUE), "\n")
cat("P(v_W3 > v_W1) =", mean(res[, 3, "velocity"] > res[, 1, "velocity"], na.rm = TRUE), "\n")
cat("P(v_W3 > v_W2) =", mean(res[, 3, "velocity"] > res[, 2, "velocity"], na.rm = TRUE), "\n")
cat("P(D_W3 > D_W1) =", mean(res[, 3, "diffusion"] > res[, 1, "diffusion"], na.rm = TRUE), "\n")

write.csv(data.frame(tree = rep(seq_along(extFiles), times = length(waveNames)),
                     wave = rep(waveNames, each = length(extFiles)),
                     velocity  = as.vector(res[, , "velocity"]),
                     diffusion = as.vector(res[, , "diffusion"]),
                     nBranches = as.vector(res[, , "nBranches"])),
          "dispersal_statistics_by_wave.csv", row.names = FALSE)

## ------------------------------- 4. wavefront distance from origin over time --
# Maximal distance of any lineage from the inferred root location, per time slice.
timeSlices <- seq(2016, 2026, by = 0.25)
wf <- matrix(NA, nrow = length(extFiles), ncol = length(timeSlices))

for (j in seq_along(extFiles)) {
  tab  <- read.csv(extFiles[j], header = TRUE)
  root <- c(tab[[COL_SLON]][which.min(tab[[COL_ST]])],
            tab[[COL_SLAT]][which.min(tab[[COL_ST]])])
  for (k in seq_along(timeSlices)) {
    ts <- timeSlices[k]
    sel <- tab[[COL_ST]] <= ts & tab[[COL_ET]] >= ts
    if (!any(sel)) next
    f <- (ts - tab[[COL_ST]][sel]) / (tab[[COL_ET]][sel] - tab[[COL_ST]][sel])
    lon <- tab[[COL_SLON]][sel] + f * (tab[[COL_ELON]][sel] - tab[[COL_SLON]][sel])
    lat <- tab[[COL_SLAT]][sel] + f * (tab[[COL_ELAT]][sel] - tab[[COL_SLAT]][sel])
    wf[j, k] <- max(geo_km(root[1], root[2], lon, lat))
  }
}
wavefront <- data.frame(
  time   = timeSlices,
  median = apply(wf, 2, median, na.rm = TRUE),
  lower  = apply(wf, 2, function(x) if (all(is.na(x))) NA else hdi(x[!is.na(x)])[1]),
  upper  = apply(wf, 2, function(x) if (all(is.na(x))) NA else hdi(x[!is.na(x)])[2]))
write.csv(wavefront, "wavefront_distance_through_time.csv", row.names = FALSE)

## ------------------------------------------- 5. seraphim's own summary stats --
# Whole-epidemic values plus dispersal statistics through time (sliding window),
# useful as an independent check on the per-wave numbers above.
spreadStatistics(localTreesDirectory, nberOfExtractionFiles = length(extFiles),
                 timeSlices = 100, onlyTipBranches = FALSE, showingPlots = TRUE,
                 outputName = "MPXV_Nigeria", nberOfCores = nberOfCores,
                 slidingWindow = 1)
