rm( list=ls() )

require( tidyverse )
require( tidymodels )
require( vip )

theme_set( theme_bw() )

data <- read_csv( "data/raw/cars2018.csv", col_types = cols() ) %>% select( -model, -model_index )

# Assuming all EDA has been done, This script tests workflows and limits processing for random Forest model
# correct mpg distribution prior to any feature engineering.

drop_cols <- c("cylinders", "exhaust_valves_per_cyl")

data_temp <- recipe( mpg ~ ., data = data ) %>%
             step_BoxCox( mpg ) %>%
             prep() %>%
             juice()


data_model <- data_temp %>% select( -drop_cols )
data_split <- initial_split( data_model, prop = .80 )
data_train <- training( data_split ) 
data_test  <- testing( data_split )

target_feature <- "mpg"
quant_features <- data_model %>% select( c(where(is.numeric), -target_feature) ) %>% names()
cat_features   <- data_model %>% select( where(is.character) ) %>% names()                                       

# no preprocessing for Randfom Forest
lm_recipe <- recipe( mpg ~ ., data = data_train ) %>%
             step_BoxCox( quant_features ) %>%
             step_center( quant_features ) %>%
             step_scale( quant_features ) %>%
             step_dummy( cat_features )

cv_splits    <- vfold_cv( data_train )
perf_metrics <- metric_set( rmse, ccc )

## Model specifications ------------------------------------------------------------------------------------------------------------------------------------

lm_model_spec <- linear_reg() %>%
  set_engine( "lm" )

rf_model_spec <- rand_forest() %>%
  set_engine( "randomForest" ) %>%
  set_mode( "regression" )

## Model workflows ------------------------------------------------------------------------------------------------------------------------------------

lm_workflow <- workflow() %>%
               add_recipe( lm_recipe ) %>%
               add_model( lm_model_spec )

rf_workflow <- workflow() %>%
               add_model( rf_model_spec ) %>%
               add_formula( mpg ~ . )

## Training: 10 fold cross validation  ------------------------------------------------------------------------------------------------------------
train_lm_model <- lm_workflow %>%
                  fit_resamples(
                    resamples = cv_splits,
                    metrics = perf_metrics,
                    control = control_resamples( save_pred = TRUE ) 
                  )

train_lm_metrics <- collect_metrics( train_lm_model ) %>% mutate( .model = "lm" )

train_rf_model <- rf_workflow %>%
                  fit_resamples(
                    resamples = cv_splits,
                    metrics = perf_metrics,
                    control = control_resamples( save_pred = TRUE ) 
                  )

train_rf_metrics <- collect_metrics( train_rf_model ) %>% mutate( .model = "rf" )

training_comparison <- bind_rows( lm_train_metrics, rf_train_metrics ) %>%
                       select( .metric, .model, n, mean, std_err ) %>%
                       arrange( .metric )


## Overall performance on test data ------------------------------------------------------------------------------------------------------------

train_lm_fit <- lm_workflow %>% fit( data_train )
train_rf_fit <- rf_workflow %>% fit( data_train ) 

lm_predictions <- train_lm_fit %>% predict( data_test )
rf_predictions <- train_rf_fit %>% predict( data_test )

test_predictions <- data_test %>% select( mpg ) %>%
                    bind_cols( lm_predictions %>% rename( lm = .pred ),
                               rf_predictions %>% rename( rf = .pred )
                             )

test_lm_metrics <- perf_metrics( test_predictions, truth = mpg, estimate = lm ) %>% mutate( .model = "lm" )
test_rf_metrics <- perf_metrics( test_predictions, truth = mpg, estimate = rf ) %>% mutate( .model = "rf" )  

test_comparison <- bind_rows( test_lm_metrics, test_rf_metrics) %>%
                   select( .metric, .model, .estimate ) %>%
                   arrange( .metric ) %>%
                   mutate( .model = recode( .model, "lm" = "Linear Regression", "rf" = "Random Forest"),
                          .label = sprintf("%s: %.3f", str_to_upper( .metric ), .estimate) )

## Visulaise Performance on test data ----------------------------------------------------------------------------------------------------------------  

p <- test_predictions %>% 
      pivot_longer(cols = c("lm", "rf"), names_to = "model", values_to = ".pred") %>%
      mutate( label = recode(model, "lm" = "Linear Regression", "rf" = "Random Forest")) %>%
      ggplot( aes(mpg, .pred) ) +
      geom_abline(lty = 2, color = "#3C99DC", alpha = .8) +
      geom_point( size = 1.5, alpha = 0.3, show.legend = FALSE ) + 
      geom_smooth( method = "lm", col = "#0F5298", fill = "#3C99DC") +
      facet_wrap( ~ label ) +
      labs( x = "Actual Values", y = "Predicted Values") +
      theme( strip.background = element_rect( fill = "#0F5298", color = "#0F5298"),
             strip.text = element_text( color = "white", size = 12))

p + geom_text( data = test_comparison %>% filter(.metric == "rmse"), aes(x = 1.51, y = 1.89, label = .label), hjust = 0) +
    geom_text( data = test_comparison %>% filter(.metric == "ccc"), aes(x = 1.51, y = 1.87, label = .label), hjust = 0) 


## check the importance of variables for each model ----------------------------------------------------------------------------------------------------------

train_lm_fit %>% pull_workflow_fit %>% 
                 vip( geom = c("point"),
                      aesthetics = list(col = "#0F5298", size = 3) ) +
                 geom_segment( aes(x = Variable, xend = Variable, y = 0, yend = Importance), col = "#0F5298" )

train_rf_fit %>% pull_workflow_fit %>%
                 vip( geom = c("point"),
                      aesthetics = list(col = "#0F5298", size = 3) ) +
                 geom_segment( aes(x = Variable, xend = Variable, y = 0, yend = Importance), col = "#0F5298" )



## Tune the random forest model ----------------------------------------------------------------------------------------------------------------------------

# update model 
tune_rf_model_spec <- rf_model_spec %>% update( mtry = tune(), min_n = tune() ) 

# update workflow
tune_rf_workflow <- rf_workflow %>% update_model( spec = tune_rf_model_spec )

tune_rf_model <- tune_rf_workflow %>% 
                 tune_grid(
                   resamples = cv_splits,
                   grid = 25,
                   metrics = perf_metrics,
                   control = control_grid( verbose = TRUE ) 
                 )

autoplot( rf_model_tune )

best_models <- tune_rf_model %>% show_best( metric = "rmse")

best_rf_model      <- tune_rf_model %>% select_best( metric = "rmse" )
best_rf_model_spec <- tune_rf_model_spec %>%
                      finalize_model( best_rf_model )

best_rf_workflow <- tune_rf_workflow %>% update_model( spec = best_rf_model_spec )

final_rf_fit <- best_rf_workflow %>%
                last_fit( split = data_split,
                          metrics = perf_metrics )

final_metrics <- collect_metrics( final_rf_fit )

final_predictions <- collect_predictions( final_rf_fit )

final_rf_fit %>% pluck(".workflow", 1) %>% pull_workflow_fit %>% vip()



rf_model_tune <- rf_model_tune_spec %>%
  tune_grid(
    preprocessor = mpg ~ .,
    resamples = cv_splits,
    grid = 25,
    metrics = perf_metrics,
    control = control_grid( verbose = TRUE ) 
  )

autoplot( rf_model_tune )

rf_model_tune %>% show_best( metric = "rmse" )

rf_best_model <- rf_model_tune %>% select_best( metric = "rmse" )

rf_final_model <- rf_model_tune_spec %>%
  finalize_model( rf_best_model ) %>%
  fit(
    mpg ~ .,
    data = model_recipe %>% juice()
  )

final_test_predictions <- model_recipe %>%
  bake( data_test ) %>% 
  select( mpg ) %>%
  bind_cols( predict( rf_final_model, new_data = model_recipe %>% bake( data_test)) %>% rename( rf = .pred) )

final_test_performance <- perf_metrics( final_test_predictions, truth = mpg, estimate = rf ) %>% mutate( .model = "rf" )  


rf_final_model %>% vip()

