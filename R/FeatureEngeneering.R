library(dplyr)

FeatureExtraction <- setRefClass("FeatureExtraction",
  fields = list(categories = "list"),
  methods = list(
    equalize_categories = function(listToEqualize) {
      min_length <- min(sapply(listToEqualize, length))
      l <- list()
      for (i in seq_along(listToEqualize)) {
        if (length(listToEqualize[[i]]) != min_length) {
          random_numbers <- sample(listToEqualize[[i]], size = min_length, replace = FALSE)
          l[[length(l) + 1]] <- random_numbers
        } else {
          l[[length(l) + 1]] <- listToEqualize[[i]]
        }
      }
      return(l)
    },
    create_default_features = function(dataset) {
      dataset$row_mean <- rowMeans(dataset, na.rm = TRUE)
      dataset$row_median <- apply(dataset, 1, median, na.rm = TRUE)
      dataset$row_min <- apply(dataset, 1, min, na.rm = TRUE)
      dataset$row_max <- apply(dataset, 1, max, na.rm = TRUE)
      dataset$row_variance <- apply(dataset, 1, var, na.rm = TRUE)
      return(dataset)
    },
    create_other_features = function(dataset) {
      dataset$Column_test_3 <- dataset$baseline_value * dataset$abnormal_short_term_variability
      dataset$Column_test_4 <- dataset$baseline_value * dataset$uterine_contractions
      dataset$Column_test_5 <- dataset$baseline_value * dataset$accelerations
      dataset$Column_test_9 <- dataset$uterine_contractions * dataset$histogram_number_of_peaks
      dataset$interaction_uterine_fetal <- dataset$uterine_contractions * dataset$fetal_movement
      dataset$interaction_movement_accelerations <- dataset$fetal_movement * dataset$accelerations
      time_window <- 3
      dataset$rolling_mean_accelerations <- rollapply(dataset$accelerations, width = time_window, mean, align = 'right', fill = NA)
      dataset$row_skewness <- apply(dataset, 1, skewness)
      dataset$row_kurtosis <- apply(dataset, 1, kurtosis)
      dataset$sum_rows <- rowSums(dataset, na.rm = TRUE)
      return(dataset)
    },
    create_categorical_feature = function(dataset, new_feature_name, column_name) {
      accelerations_min <- min(dataset[[column_name]], na.rm = TRUE)
      accelerations_max <- max(dataset[[column_name]], na.rm = TRUE)
      accelerations_interval <- (accelerations_max - accelerations_min) / length(FeatureExtraction$categories)
      bins <- seq(accelerations_min, accelerations_max, by = accelerations_interval)
      dataset[[new_feature_name]] <- cut(dataset[[column_name]], breaks = bins, labels = names(FeatureExtraction$categories), include.lowest = TRUE)
      dataset[[new_feature_name]] <- as.integer(factor(dataset[[new_feature_name]]))
      return(dataset)
    },
    create_features = function(dataset, targets) {
      dataset_targets <- list()
      for (target in targets) {
        dataset_targets[[target]] <- dataset[[target]]
        dataset[[target]] <- NULL
      }
      dataset <- create_default_features(dataset)
      dataset <- create_other_features(dataset)
      dataset <- create_categorical_feature(dataset, "accelerations_category", "accelerations")
      dataset <- create_categorical_feature(dataset, "fetal_movement_category", "fetal_movement")
      for (key in names(dataset_targets)) {
        dataset[[key]] <- dataset_targets[[key]]
      }
      return(dataset)
    }
  )
)

FeatureSelector <- setRefClass("FeatureSelector",
  fields = list(data = "data.frame", labels = "factor"),
  methods = list(
    select_features_mrmr = function(k = 5) {
      return(mRMR.classif(data, labels, k))
    },
    select_features_sequential = function(k = 5) {
      control <- rfeControl(functions=rfFuncs, method="cv", number=10)
      results <- rfe(data, labels, sizes=c(k), rfeControl=control)
      return(names(results$optVariables))
    }
  )
)