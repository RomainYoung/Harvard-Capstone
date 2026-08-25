##########################################################
# MovieLens Capstone - HarvardX PH125.9x
# Author: Romain
##########################################################
#
# Goal: predict movie ratings on the MovieLens 10M dataset and get the
# RMSE as low as I reasonably can with the tools from this course.
#
# I started with the bias-based approach from the ML course (movie +
# user effects, then regularized), which gets to about 0.865. That's
# fine but it's basically copying the course material, so I pushed
# further with matrix factorization (recosystem package) which gets
# down to ~0.78. That's the model I'm submitting.
#
# Note on runtime: the full thing (download + all models) takes maybe
# 10-15 min on a normal laptop, most of it the lambda search and the
# recosystem training. Nothing crazy but don't run it 5 min before the
# deadline.

## ------------------------------------------------------
## 0. Packages
## ------------------------------------------------------

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(recosystem)

## ------------------------------------------------------
## 1. Create edx and final_holdout_test
##    (official code from the course, unchanged)
## ------------------------------------------------------

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                          stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

## ------------------------------------------------------
## 2. Quick look at the data
##    (this is basically what the quiz on the course site asks about,
##    keeping it here since it's useful context for the report too)
## ------------------------------------------------------

dim(edx)
n_distinct(edx$movieId)
n_distinct(edx$userId)

edx %>% count(rating) %>% arrange(desc(n))

edx %>%
  group_by(title) %>%
  summarize(n = n(), .groups = "drop") %>%
  arrange(desc(n)) %>%
  slice(1:10)

# ratings distribution - whole stars clearly more common than half stars
edx %>%
  mutate(half_star = rating %% 1 != 0) %>%
  count(half_star)

## ------------------------------------------------------
## 3. RMSE function
## ------------------------------------------------------

RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

## ------------------------------------------------------
## 4. Train/test split inside edx
##    final_holdout_test doesn't get touched until the very end
## ------------------------------------------------------

set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

test_set <- temp %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

rm(test_index, temp, removed)

## ------------------------------------------------------
## 5. Baseline models: mean, movie effect, movie + user effect
##    This is the standard approach from the ML course, I'm using it
##    as a stepping stone / point of comparison, not as the final model.
## ------------------------------------------------------

mu <- mean(train_set$rating)

rmse_results <- tibble(method = "Just the average", RMSE = RMSE(test_set$rating, mu))

movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

pred_movie <- mu + test_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  pull(b_i)

rmse_results <- rmse_results %>%
  add_row(method = "Movie effect", RMSE = RMSE(test_set$rating, pred_movie))

user_avgs <- train_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

pred_movie_user <- test_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

rmse_results <- rmse_results %>%
  add_row(method = "Movie + user effect", RMSE = RMSE(test_set$rating, pred_movie_user))

rmse_results

## ------------------------------------------------------
## 6. Regularization
##    Shrinks b_i and b_u for movies/users with very few ratings, since
##    those estimates are noisy. Looking for the lambda that minimizes
##    RMSE on test_set.
## ------------------------------------------------------

lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l) {

  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + l))

  b_u <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu) / (n() + l))

  pred <- test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)

  RMSE(test_set$rating, pred)
})

qplot(lambdas, rmses)

lambda_opt <- lambdas[which.min(rmses)]
lambda_opt   # 4.75 when I ran it

rmse_results <- rmse_results %>%
  add_row(method = paste0("Regularized movie + user (lambda = ", lambda_opt, ")"),
          RMSE = min(rmses))

rmse_results
# Regularization barely moves the needle here (0.866 -> 0.865). Makes
# sense - most movies/users in this dataset already have plenty of
# ratings, so there isn't much noise to shrink away. This confirms the
# bias-based approach is close to its ceiling, which is why I went with
# matrix factorization for the actual final model below.

## ------------------------------------------------------
## 7. Matrix factorization with recosystem
##    This is the part that actually goes beyond the course material.
##    Instead of two additive bias terms, it learns a low-rank latent
##    factor representation for users and movies (like SVD, but fit
##    with SGD so it handles the missing entries in the ratings matrix
##    without needing to impute anything).
## ------------------------------------------------------

train_reco <- with(train_set, data_memory(user_index = userId,
                                           item_index = movieId,
                                           rating = rating))
test_reco <- with(test_set, data_memory(user_index = userId,
                                         item_index = movieId,
                                         rating = rating))

# I tried recosystem's built-in tune() first but it runs a 5-fold CV
# internally which made it way too slow for my machine. Doing a small
# manual grid search against test_set instead - not quite as thorough
# but good enough, and much faster.

grid <- expand.grid(dim = c(10, 20, 30),
                     costp_l2 = c(0.01, 0.1),
                     costq_l2 = c(0.01, 0.1))

grid_rmse <- numeric(nrow(grid))

for (i in seq_len(nrow(grid))) {
  set.seed(1)
  r <- Reco()
  r$train(train_reco, opts = list(dim = grid$dim[i],
                                   lrate = 0.1,
                                   costp_l2 = grid$costp_l2[i],
                                   costq_l2 = grid$costq_l2[i],
                                   nthread = 1,
                                   niter = 20,
                                   verbose = FALSE))
  pred <- r$predict(test_reco, out_memory())
  grid_rmse[i] <- RMSE(test_set$rating, pred)
}

grid$RMSE <- grid_rmse
grid %>% arrange(RMSE)

# Best region was around dim=30, costp_l2=0.01, costq_l2=0.1. Nudged it
# a bit further (higher dim, more iterations) and landed on this:

set.seed(1)
r_test <- Reco()
r_test$train(train_reco, opts = list(dim = 40, lrate = 0.1,
                                      costp_l2 = 0.01, costq_l2 = 0.10,
                                      nthread = 1, niter = 30, verbose = FALSE))
pred_reco <- r_test$predict(test_reco, out_memory())
rmse_reco <- RMSE(test_set$rating, pred_reco)
rmse_reco   # ~0.788 on test_set

rmse_results <- rmse_results %>%
  add_row(method = "Matrix factorization (recosystem, dim=40)", RMSE = rmse_reco)

rmse_results

## ------------------------------------------------------
## 8. Final model: train on all of edx, evaluate on final_holdout_test
##    ONE TIME ONLY - this is the number that counts.
## ------------------------------------------------------

edx_reco <- with(edx, data_memory(user_index = userId,
                                   item_index = movieId,
                                   rating = rating))
final_reco <- with(final_holdout_test, data_memory(user_index = userId,
                                                     item_index = movieId,
                                                     rating = rating))

set.seed(1)
r_final <- Reco()
r_final$train(edx_reco, opts = list(dim = 40, lrate = 0.1,
                                     costp_l2 = 0.01, costq_l2 = 0.10,
                                     nthread = 1, niter = 30, verbose = TRUE))

predicted_ratings <- r_final$predict(final_reco, out_memory())

final_rmse <- RMSE(final_holdout_test$rating, predicted_ratings)

cat("Final RMSE on final_holdout_test:", final_rmse, "\n")

## ------------------------------------------------------
## 9. Summary
## ------------------------------------------------------

rmse_results <- rmse_results %>%
  add_row(method = "FINAL MODEL - evaluated on final_holdout_test", RMSE = final_rmse)

print(rmse_results, n = Inf)
