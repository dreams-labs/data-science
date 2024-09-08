# Modeling Strategy

### Overview

This project aims to predict cryptocurrency price movements using wallet transaction data. To ensure the model generalizes well to unseen data, we’ve structured the data into distinct time periods and evaluation sets.

### Dataset Structure

Data will be segmented into four sets:
* **Training set:** used to train the model
* **Validation set:** used to tune model parameters
* **Test set:** used to assess whether model generalizes within the same time period as the Training and Validation sets
* **Future set:** used to assess whether the model generalizes within a different time period from the Training, Validation, and Test sets. 

## Time Periods

It will be very important to strictly segment price data into different periods, because prices are used for feature engineering, model construction, model validation, and model generalization. If price movements from one of these processes were also included in another, data leakaage would contaminate model performance and likely render it unable to generalize its performance. 

As such, price movements will be split into three periods:

1. **Progeny Period** (e.g. price movements up to 3/1/24)
   - Price movements during this period will be used for feature engineering. This ensures that the model only relies on information that was available at the time of the prediction.

2. **Modeling Period** (e.g. 3/1/24–5/1/24):
   - Price movements during this period will generate the target variables in the Training, Validation, and Test sets. 
   - Price movements during this period will also be used to build features for the Future set since it will have access to all transactions up until the start of the Future period. 

3. **Future Period** (e.g. 5/1/24-7/1/24):
   - Price movements during this period will generate the target variables in the Future set to assess whether the model can generalize beyond the original time frame, particularly in future market conditions that may be significantly different from the past.

### Evaluation Strategy

- **Validation and Test Sets**: These sets assess whether the model can generalize within the training period. Strong performance here would indicate that the model is able to learn meaningful patterns without overfitting.
- **Future Set**: Performance on this set will determine if the model can generalize to other time periods, particularly when market conditions change.


# Testing

Tests are built in the tests/ directory and configured with pytest.ini. 

## Running Tests

Tests can be initiated from the main data-science directory with the standard command >`pytest`. 

There is a `slow` flag applied to slow tests that involve large dataframes or queries. To ignore these but run other tests, use the command >`pytest -m "not slow"`. 