# https://juliasilge.com/blog/lego-sets/

library(tidyverse)
lego_sets <- read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2022/2022-09-06/sets.csv.gz')
glimpse(lego_sets)


library(tidymodels)

set.seed(123)
lego_split <- lego_sets %>%
  filter(num_parts > 0) %>%
  transmute(num_parts = log10(num_parts), name) %>%
  initial_split(strata = num_parts)

lego_train <- training(lego_split)
lego_test <- testing(lego_split)

set.seed(234)
lego_folds <- vfold_cv(lego_train, strata = num_parts)
lego_folds


library(textrecipes)

lego_rec <- recipe(num_parts ~ name, data = lego_train) %>%
  step_tokenize(name) %>%
  step_tokenfilter(name, max_tokens = 200) %>%
  step_tfidf(name)

lego_rec

svm_spec <- svm_linear(mode = "regression")
lego_wf <- workflow(lego_rec, svm_spec)

set.seed(234)

doParallel::registerDoParallel()
lego_rs <- fit_resamples(lego_wf, lego_folds)
collect_metrics(lego_rs)

final_fitted <- last_fit(lego_wf, lego_split)
collect_metrics(final_fitted)


final_fitted %>%
  extract_workflow() %>%
  tidy() %>%
  arrange(-estimate)


library(vetiver)

v <- final_fitted %>%
  extract_workflow() %>%
  vetiver_model(model_name = "lego-sets")
v

library(pins)
board <- board_folder("Boards/")
board %>% vetiver_pin_write(v)


vetiver_write_plumber(board, "lego-sets", rsconnect = FALSE)
vetiver_write_docker(v)
