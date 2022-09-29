rm( list=ls() )

require( tidyverse )
require( tidymodels )
require( vip )

theme_set( theme_bw() )

data <- read_csv( "data/raw/cars2018.csv", col_types = cols() ) %>% select( -model, -model_index )

glimpse( data )
summary( data )

target_feature <- "mpg"
quant_features <- data %>% select( c(where(is.numeric), -target_feature) ) %>% names()
cat_features   <- data %>% select( where(is.character) ) %>% names()                                       

ggplot(data, aes(x = mpg)) +
  geom_histogram( bins = 25, col = "#0F5298", fill = "#3C99DC", alpha = .8) +
  labs(x = "Fuel efficiency (mpg)",
       y = "Number of cars") 

ggplot(data, aes(x = mpg)) +
  geom_histogram( bins = 25, col = "#0F5298", fill = "#3C99DC", alpha = .8) +
  facet_grid( aspiration ~ transmission ) +
  labs(x = "Fuel efficiency (mpg)",
          y = "Number of cars") 


# Find which columns contain missing values
check_missing <- data %>% summarise( across( .fns = ~any( is.na(.x) ))) %>% pivot_longer( cols = everything(), values_to = "is_missing" )
any_missing   <- check_missing %>% pull( is_missing ) %>% any()
  
if ( any_missing ) {
  cols_missing <- check_missing %>% filter( is_missing ) %>% pull( name )
  data %>% select( cols_missing ) %>% visdat::vis_miss( cluster = TRUE )
  
  # which imputation to use?
}

# Check for variables with near zero variance
check_nzv <- data %>% caret::nearZeroVar( saveMetrics = TRUE ) %>% tibble::rownames_to_column() 
any_nzv   <- check_nzv %>% pull( nzv ) %>% any() 

if ( any_nzv ){
  
  # add step_nzv to recipe
}

# check correlations between continuous features
quant_preds <- recipe( mpg ~., data = data ) %>%
               step_BoxCox( quant_features ) %>%
               step_center( quant_features ) %>%
               step_scale( quant_features ) %>%
               prep() %>%
               juice() %>% 
               select( quant_features )

quant_corr  <- cor( quant_preds ) 

corrplot::corrplot(quant_corr, addgrid.col = rgb(0, 0, 0, .05), order = "hclust", diag = FALSE, tl.col = "black", tl.cex = .8, 
                   addCoef.col = "black", number.cex = .5, addrect = 3, mar = c(2,2,2,2), xpd = TRUE )

# based on corr plot suggest dropping cylinders (displacement has more unique values) and either exhaust valves or intake valves
drop_cols <- c("cylinders", "exhaust_valves_per_cyl")

data_model <- data %>% select( -drop_cols )
data_split <- initial_split( data_model, prop = .80 )
data_train <- training( data_split ) 
data_test  <- testing( data_split )

target_feature <- "mpg"
quant_features <- data_model %>% select( c(where(is.numeric), -target_feature) ) %>% names()
cat_features   <- data_model %>% select( where(is.character) ) %>% names()                                       

model_recipe <- recipe( mpg ~ ., data = data_train ) %>%
                step_BoxCox( target_feature ) %>% 
                step_BoxCox( quant_features ) %>%
                step_center( quant_features ) %>%
                step_scale( quant_features ) %>%
                step_dummy( cat_features ) %>%
                # take care of the preparation here
                prep( training = data_train, retain = TRUE )

cv_splits <- vfold_cv( model_recipe %>% juice() )

perf_metrics <- metric_set( rmse, ccc )

## Model specifications ------------------------------------------------------------------------------------------------------------------------------------

lm_model_spec <- linear_reg() %>%
                 set_engine( "lm" )

rf_model_spec <- rand_forest() %>%
                 set_engine( "randomForest" ) %>%
                 set_mode( "regression" )

## CV training for each model ------------------------------------------------------------------------------------------------------------
lm_model_fits <- lm_model_spec %>% 
                 fit_resamples( preprocessor = mpg ~ .,
                                resamples = cv_splits, 
                                metrics = perf_metrics,
                                control = control_resamples( save_pred = TRUE ) )

lm_train_metrics <- collect_metrics( lm_model_fits ) %>% mutate( .model = "lm" )

rf_model_fits <- rf_model_spec %>% 
                 fit_resamples( preprocessor = mpg ~ .,
                                resamples = cv_splits, 
                                metrics = perf_metrics,
                                control = control_resamples( save_pred = TRUE ) )

rf_train_metrics <- collect_metrics( rf_model_fits ) %>% mutate( .model = "rf" )

cv_training_metrics <- bind_rows( lm_train_metrics, rf_train_metrics ) %>%
                       select( .metric, .model, n, mean, std_err ) %>%
                       arrange( .metric )


## Overall fit to training data ------------------------------------------------------------------------------------------------------------

lm_full_fits <- lm_model_spec %>%
                fit(
                  mpg ~ .,
                  data = model_recipe %>% juice()
                )

rf_full_fits <- rf_model_spec %>%
                fit(
                  mpg ~ .,
                  data = model_recipe %>% juice()
                )

## Performance on test data ----------------------------------------------------------------------------------------------------------------  

test_predictions <- model_recipe %>%
                    bake( data_test ) %>% 
                    select( mpg ) %>%
                    bind_cols( predict( lm_full_fits, new_data = model_recipe %>% bake( data_test)) %>% rename( lm = .pred ) ) %>%
                    bind_cols( predict( rf_full_fits, new_data = model_recipe %>% bake( data_test)) %>% rename( rf = .pred) )

test_performance <- bind_rows(
                      perf_metrics( test_predictions, truth = mpg, estimate = lm ) %>% mutate( .model = "lm" ),
                      perf_metrics( test_predictions, truth = mpg, estimate = rf ) %>% mutate( .model = "rf" )  
                    ) %>%
                    select( .metric, .model, .estimate ) %>%
                    arrange( .metric ) %>%
                    mutate( label = recode( .model, "lm" = "Linear Regression", "rf" = "Random Forest"),
                            .label = sprintf("%s: %.3f", str_to_upper( .metric ), .estimate) )

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


#test_performance %>% mutate( .label = sprintf("%s:\t%.3f", str_to_upper( .metric ), .estimate) )

p + geom_text( data = test_performance %>% filter(.metric == "rmse"), aes(x = 1.51, y = 1.89, label = .label), hjust = 0) +
    geom_text( data = test_performance %>% filter(.metric == "ccc"), aes(x = 1.51, y = 1.87, label = .label), hjust = 0) 


## check the importance of variables for each model ----------------------------------------------------------------------------------------------------------

lm_full_fits %>% vip(
  geom = c("point"),
  aesthetics = list(col = "#0F5298", size = 3)
) + geom_segment(aes(x = Variable, xend = Variable, y = 0, yend = Importance), col = "#0F5298")


rf_full_fits %>% vip(
  geom = c("point"),
  aesthetics = list(col = "#0F5298", size = 3)
) + geom_segment(aes(x = Variable, xend = Variable, y = 0, yend = Importance), col = "#0F5298")


## Tune the random forest model
rf_model_tune_spec <- rand_forest( mtry = tune(), min_n = tune(), trees = 1000 ) %>%
                      set_engine( "randomForest" ) %>%
                      set_mode( "regression" )

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

