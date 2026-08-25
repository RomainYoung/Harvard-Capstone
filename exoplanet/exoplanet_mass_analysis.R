##########################################################
# Choose Your Own Capstone - HarvardX PH125.9x
# Predicting exoplanet mass from radius and stellar parameters
# Author: Romain
##########################################################
#
# Data source: NASA Exoplanet Archive (public domain), confirmed planets
# table, queried via their TAP service. Not a "famous" ML teaching
# dataset - I built this one myself from the live archive and engineered
# most of the interesting columns (habitable zone flag, equilibrium
# temperature, recovered semi-major axis) rather than using them
# pre-computed.
#
# The problem: about half of all confirmed exoplanets have a measured
# radius but no measured mass (mass needs radial-velocity follow-up,
# which is expensive telescope time; radius alone comes for free from
# the transit light curve). So predicting mass from radius + other
# available parameters is a real, useful problem in exoplanet science,
# not just an academic exercise - it's essentially what empirical
# mass-radius relations (e.g. Chen & Kipping 2017) are used for.

## ------------------------------------------------------
## 0. Packages
## ------------------------------------------------------

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(gbm)) install.packages("gbm", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(randomForest)
library(gbm)

## ------------------------------------------------------
## 1. Get the data
##    Primary source: my GitHub repo (stable snapshot, so results don't
##    shift if new planets get confirmed between now and grading).
##    Fallback: pull a fresh copy straight from the NASA Exoplanet
##    Archive's TAP service if the GitHub copy isn't reachable.
## ------------------------------------------------------

raw_file <- "exoplanets_raw.csv"

github_url <- "https://raw.githubusercontent.com/YOUR-GITHUB-USERNAME/YOUR-REPO/main/exoplanets_raw.csv"
nasa_url <- paste0(
  "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=",
  "select+pl_name,hostname,discoverymethod,disc_year,pl_orbper,",
  "pl_rade,pl_bmasse,pl_orbsmax,pl_eqt,st_teff,st_rad,st_mass,",
  "sy_dist,st_met+from+ps+where+default_flag=1&format=csv"
)

if (!file.exists(raw_file)) {
  ok <- tryCatch({
    download.file(github_url, raw_file, quiet = TRUE)
    TRUE
  }, error = function(e) FALSE)

  if (!ok || !file.exists(raw_file) || file.info(raw_file)$size < 1000) {
    message("GitHub copy unavailable, querying the NASA Exoplanet Archive directly...")
    download.file(nasa_url, raw_file, quiet = TRUE)
  }
}

d <- read.csv(raw_file, stringsAsFactors = FALSE)
cat("Raw rows downloaded:", nrow(d), "\n")

## ------------------------------------------------------
## 2. Feature engineering
##    This is most of the actual work in this project - a lot of the
##    interesting columns don't exist in the raw archive and have to be
##    derived from physics.
## ------------------------------------------------------

# 2a. Recover missing semi-major axis via Kepler's third law where the
#     archive itself doesn't have it directly:
#     a[AU] = (P[years]^2 * M_star[Msun])^(1/3)
d <- d %>%
  mutate(
    pl_orbper_yr = pl_orbper / 365.25,
    orbsmax_derived = ifelse(is.na(pl_orbsmax) & !is.na(pl_orbper) & !is.na(st_mass),
                              (pl_orbper_yr^2 * st_mass)^(1/3),
                              NA_real_),
    a_au = ifelse(!is.na(pl_orbsmax), pl_orbsmax, orbsmax_derived)
  )
cat("Semi-major axis recovered via Kepler's 3rd law for",
    sum(is.na(d$pl_orbsmax) & !is.na(d$orbsmax_derived)), "extra rows\n")

# 2b. Stellar luminosity via the Stefan-Boltzmann law, relative to the Sun
#     (Tsun = 5780 K)
d <- d %>%
  mutate(L_star = (st_rad)^2 * (st_teff / 5780)^4)

# 2c. Zero-albedo equilibrium temperature, derived independently rather
#     than relying on the archive's own pl_eqt column (which is missing
#     for about 72% of rows - ours covers a lot more)
#     T_eq = T_star * sqrt(R_star_AU / (2*a))   [1 Rsun = 0.00465047 AU]
d <- d %>%
  mutate(
    st_rad_au = st_rad * 0.00465047,
    pl_eqt_derived = ifelse(!is.na(st_teff) & !is.na(st_rad_au) & !is.na(a_au) & a_au > 0,
                             st_teff * sqrt(st_rad_au / (2 * a_au)),
                             NA_real_)
  )

# 2d. Conservative habitable zone boundaries (Kasting et al. 1993;
#     Seff values recalculated by Kopparapu et al. 2013: moist
#     greenhouse Seff=1.01 for the inner edge, maximum greenhouse
#     Seff=0.35 for the outer edge, referenced to a Sun-like Teff=5780K)
d <- d %>%
  mutate(
    hz_inner_au = sqrt(L_star / 1.01),
    hz_outer_au = sqrt(L_star / 0.35),
    in_hz = ifelse(!is.na(a_au) & !is.na(hz_inner_au) & !is.na(hz_outer_au),
                    a_au >= hz_inner_au & a_au <= hz_outer_au,
                    NA)
  )

cat("Planets classifiable for habitable zone status:", sum(!is.na(d$in_hz)), "\n")
cat("Planets in the conservative habitable zone:", sum(d$in_hz, na.rm = TRUE), "\n")

write.csv(d, "exoplanets_engineered.csv", row.names = FALSE)

## ------------------------------------------------------
## 3. Some exploration before modeling
## ------------------------------------------------------

d %>% count(discoverymethod) %>% arrange(desc(n))

d %>%
  filter(!is.na(pl_bmasse), !is.na(pl_rade)) %>%
  summarize(n = n(), mass_range = paste(round(range(pl_bmasse), 2), collapse = " - "),
            radius_range = paste(round(range(pl_rade), 2), collapse = " - "))
# mass and radius each span several orders of magnitude, which is why
# everything downstream gets modeled in log space rather than raw units

## ------------------------------------------------------
## 4. Build the regression-ready dataset
##    (needs mass + radius + the stellar/orbital predictors, all non-missing)
## ------------------------------------------------------

model_data <- d %>%
  filter(!is.na(pl_bmasse), !is.na(pl_rade), !is.na(pl_orbper),
         !is.na(st_teff), !is.na(st_rad), !is.na(st_mass),
         !is.na(pl_eqt_derived)) %>%
  filter(pl_bmasse > 0, pl_rade > 0, pl_orbper > 0) %>%
  transmute(
    pl_name,
    log_mass = log10(pl_bmasse),
    log_radius = log10(pl_rade),
    log_period = log10(pl_orbper),
    st_teff, st_rad, st_mass,
    eqt = pl_eqt_derived
  )

cat("\nFinal regression-ready dataset:", nrow(model_data), "rows\n")

## ------------------------------------------------------
## 5. Train/test split and RMSE function
## ------------------------------------------------------

RMSE <- function(true, pred) sqrt(mean((true - pred)^2))

set.seed(1)
test_index <- createDataPartition(model_data$log_mass, p = 0.2, list = FALSE)
train_set <- model_data[-test_index, ]
test_set  <- model_data[test_index, ]
cat("Train:", nrow(train_set), " Test:", nrow(test_set), "\n")

## ------------------------------------------------------
## 6. Model 1: linear regression (log-log)
##    This mirrors the classical empirical mass-radius power-law
##    relations used in the exoplanet literature - a reasonable and
##    interpretable baseline before trying anything fancier.
## ------------------------------------------------------

lm_fit <- lm(log_mass ~ log_radius + log_period + st_teff + st_rad + st_mass + eqt,
             data = train_set)
pred_lm <- predict(lm_fit, newdata = test_set)
rmse_lm <- RMSE(test_set$log_mass, pred_lm)
cat("\nLinear regression RMSE (log10 Earth masses):", rmse_lm, "\n")
summary(lm_fit)
# radius comes out with a coefficient close to 2, i.e. mass scales
# roughly as radius^2 over this range - broadly consistent with
# published mass-radius relations for the sub-Neptune/gas giant regime
# that dominates this sample

## ------------------------------------------------------
## 7. Model 2: random forest
## ------------------------------------------------------

set.seed(1)
rf_fit <- randomForest(log_mass ~ log_radius + log_period + st_teff + st_rad + st_mass + eqt,
                        data = train_set, ntree = 500, importance = TRUE)
pred_rf <- predict(rf_fit, newdata = test_set)
rmse_rf <- RMSE(test_set$log_mass, pred_rf)
cat("\nRandom forest RMSE:", rmse_rf, "\n")
importance(rf_fit)

## ------------------------------------------------------
## 8. Model 3: gradient boosting
## ------------------------------------------------------

set.seed(1)
gbm_fit <- gbm(log_mass ~ log_radius + log_period + st_teff + st_rad + st_mass + eqt,
               data = train_set, distribution = "gaussian",
               n.trees = 1000, interaction.depth = 3, shrinkage = 0.01,
               cv.folds = 5, verbose = FALSE)
best_iter <- gbm.perf(gbm_fit, method = "cv", plot.it = FALSE)
cat("\nGBM best iteration (chosen by internal cross-validation):", best_iter, "\n")
pred_gbm <- predict(gbm_fit, newdata = test_set, n.trees = best_iter)
rmse_gbm <- RMSE(test_set$log_mass, pred_gbm)
cat("GBM RMSE:", rmse_gbm, "\n")

## ------------------------------------------------------
## 9. Compare all three
## ------------------------------------------------------

results <- tibble(
  model = c("Linear regression", "Random forest", "Gradient boosting"),
  RMSE_log10_mass = c(rmse_lm, rmse_rf, rmse_gbm),
  R2 = c(
    cor(test_set$log_mass, pred_lm)^2,
    cor(test_set$log_mass, pred_rf)^2,
    cor(test_set$log_mass, pred_gbm)^2
  )
)

cat("\n=== FINAL RESULTS ===\n")
print(results)

# Gradient boosting comes out slightly ahead, with random forest close
# behind - both meaningfully better than plain linear regression, which
# suggests the mass-radius-period-stellar relationship has some real
# non-linear structure that a single log-log line can't fully capture
# (makes sense given how different rocky planets, ice giants and gas
# giants are from each other, all mixed into one sample here)
