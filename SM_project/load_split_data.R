load_split_data <- function(dataset_loc) {
  # Load the dataset train and test splits. Returns train, test, and the whole
  # df. Input to the function is the path to the dataset.
  set.seed(42)
  load(dataset_loc)
  df <- ex3.health
  sample <- sample(c(TRUE, FALSE), nrow(df), replace=TRUE, prob=c(0.8,0.2))
  train  <- df[sample, ]
  test   <- df[!sample, ]
  return(list("train"=train, "test"=test, "df"=df))
}