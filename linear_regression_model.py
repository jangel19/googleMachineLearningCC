#data
import numpy as np
import pandas as pd

#machine learning stuff
import keras
import ml_edu.experiment
import ml_edu.results

#data vis
import plotly.express as px

chicago_taxi_dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")

#updating dataframe to use specific info
training_df = chicago_taxi_dataset.loc[:, ('TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE')]

print('reading dataset successfully')
print('total nums of rows: {0}\n\n'.format(len(training_df.index)))
#training_df.head(200)
training_df.describe(include='all')

#getting the max fare
max_fare = training_df['FARE'].max()
print("what is the max fare?        Answer: ${fare:.2f}".format(fare = max_fare))

#getting mean dist
mean_dist = training_df['TRIP_MILES'].mean()
print("what is the mean of the dist?                Answer: {mean:.4f} miles".format(mean = mean_dist))

#getting amount of cab companies
diff_companies = training_df['COMPANY'].nunique()
print("what is the num of companies?            answer: {number}".format(number = diff_companies))

#most used payment
top_payment = training_df['PAYMENT_TYPE'].value_counts().idxmax()
print("what is the most used payment?           answer:{type}".format(type = top_payment))

#find if any features missing
missing_vals = training_df.isnull().sum().sum()
print("are there any missing functions?         answer", "no" if missing_vals == 0 else "yes")


#CORRELATION MATRIX
training_df.corr(numeric_only = True)

#view pairplot
fig = px.scatter_matrix(training_df, dimensions=["FARE", "TRIP_MILES", "TRIP_SECONDS"])
#for all of the graphs since i am using plotly
#do not get stored in files and go to ide when shown
#do not want to hassle as i can see them in google colab

#TRAINING TIMEEE HEH

def create_model(
        settings: ml_edu.experiment.ExperimentSettings,
        metrics: list[keras.metrics.Metric],
)->keras.Model:
    """Create and compile a simple linear regression model."""
    # Describe the topography of the model.
    # The topography of a simple linear regression model
    # is a single node in a single layer.
    inputs = {name: keras.Input(shape=(1,), name=name) for name in settings.input_features}
    concatenated_inputs = keras.layers.Concatenate()(list(inputs.values()))
    outputs = keras.layers.Dense(units=1)(concatenated_inputs)
    model = keras.Model(inputs = inputs, outputs =outputs)

    #comple the model topography into code so keras can run
    #config training to minnimize the models mean sqd error
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=settings.learning_rate),
                  loss="mean_squared_error",
                  metrics = metrics
    )

    return model

def train_model(experiment_name: str, model: keras.Model,
        dataset: pd.DataFrame,
        label_name: str,
        settings: ml_edu.experiment.ExperimentSettings,
)-> ml_edu.experiment.Experiment:
    """train the model by feeding it data"""

    #feed model the feature and label and the model will train for the spec num of epochs
    features = {name: dataset[name].values for name in settings.input_features}
    label = dataset[label_name].values
    history = model.fit(x= features,
                        y = label,
                        batch_size = settings.batch_size,
                        epochs = settings.number_epochs)
    return ml_edu.experiment.Experiment(
        name = experiment_name,
        settings = settings,
        model = model,
        epochs= history.epoch,
        metrics_history=pd.DataFrame(history.history)
    )

print("success linear reg functions complete")

# The following variables are the hyperparameters.
settings_1 = ml_edu.experiment.ExperimentSettings(
    learning_rate = 1.0, # when the learning rate increased the prediction line is less accurate but still ok
    number_epochs = 20,
    batch_size = 50,
    input_features = ['TRIP_MILES']
)

metrics = [keras.metrics.RootMeanSquaredError(name='rmse')]

model_1 = create_model(settings_1, metrics)

experiment_1 = train_model('one_feature', model_1, training_df, 'FARE', settings_1)

ml_edu.results.plot_experiment_metrics(experiment_1, ['rmse'])
ml_edu.results.plot_model_predictions(experiment_1, training_df, 'FARE')
#after running this it took ~five epochs to converge on the final model
#the mdoel fits the data very good

settings_2 = ml_edu.experiment.ExperimentSettings(
    learning_rate = 0.0001,
    number_epochs = 20,
    batch_size = 50,
    input_features = ['TRIP_MILES']
)

metrics = [keras.metrics.RootMeanSquaredError(name='rmse')]

model_2 = create_model(settings_2, metrics)

experiment_2 = train_model('one_feature_hyper', model_2, training_df, 'FARE', settings_2)

ml_edu.results.plot_experiment_metrics(experiment_2, ['rmse'])
ml_edu.results.plot_model_predictions(experiment_2, training_df, 'FARE')
#prediction line is very off


settings_3 = ml_edu.experiment.ExperimentSettings(
    learning_rate = 0.001,
    number_epochs = 20,
    batch_size = 500,
    input_features = ['TRIP_MILES', 'TRIP_MINUTES']
)

training_df['TRIP_MINUTES'] = training_df['TRIP_SECONDS']/60 #now we r using two features to train model

metrics = [keras.metrics.RootMeanSquaredError(name='rmse')]

model_3 = create_model(settings_3, metrics)

experiment_3 = train_model('two_features', model_3, training_df, 'FARE', settings_3)

ml_edu.results.plot_experiment_metrics(experiment_3, ['rmse'])
ml_edu.results.plot_model_predictions(experiment_3, training_df, 'FARE')
#this one with the higher batch size ran faster
#if the RMSE for the model trained with one feature was
#3.7457 and the RMSE for the model with two features is 3.4787, that means that
#on average the model with two features makes predictions that are about $0.27
#closer to the observed fare.
# When training a model with more than one feature, it is important that all
# numeric values are roughly on the same scale. In this case, TRIP_SECONDS and
# TRIP_MILES do not meet this criteria. The mean value for TRIP_MILES is 8.3 and
# the mean for TRIP_SECONDS is 1,320; that is two orders of magnitude difference.
# In contrast, the mean for TRIP_MINUTES is 22, which is more similar to the scale
# of TRIP_MILES (8.3) than TRIP_SECONDS (1,320). Of course, this is not the
# only way to scale values before training, but you will learn about that in
# another module.
# How well do you think the model comes to the ground truth fare calculation for
# Chicago taxi trips?
# -----------------------------------------------------------------------------
# answer = '''
# In reality, Chicago taxi cabs use a documented formula to determine cab fares.
# For a single passenger paying cash, the fare is calculated like this:

# FARE = 2.25 * TRIP_MILES + 0.12 * TRIP_MINUTES + 3.25

# Typically with machine learning problems you would not know the 'correct'
# formula, but in this case you can use this knowledge to evaluate your model.
# Take a look at your model output (the weights and bias) and determine how
# well it matches the ground truth fare calculation. You should find that the
# model is roughly close to this formula.

ml_edu.results.compare_experiment([experiment_1, experiment_3], ['rmse'], training_df, training_df['FARE'].values)#compare models

#defining functions to make predictions


def format_currency(x):
  return "${:.2f}".format(x)

def build_batch(df, batch_size):
  batch = df.sample(n=batch_size).copy()
  batch.set_index(np.arange(batch_size), inplace=True)
  return batch

def predict_fare(model, df, features, label, batch_size=50):
  batch = build_batch(df, batch_size)
  predicted_values = model.predict_on_batch(x={name: batch[name].values for name in features})

  data = {"PREDICTED_FARE": [], "OBSERVED_FARE": [], "L1_LOSS": [],
          features[0]: [], features[1]: []}
  for i in range(batch_size):
    predicted = predicted_values[i][0]
    observed = batch.at[i, label]
    data["PREDICTED_FARE"].append(format_currency(predicted))
    data["OBSERVED_FARE"].append(format_currency(observed))
    data["L1_LOSS"].append(format_currency(abs(observed - predicted)))
    data[features[0]].append(batch.at[i, features[0]])
    data[features[1]].append("{:.2f}".format(batch.at[i, features[1]]))

  output_df = pd.DataFrame(data)
  return output_df

def show_predictions(output):
  header = "-" * 80
  banner = header + "\n" + "|" + "PREDICTIONS".center(78) + "|" + "\n" + header
  print(banner)
  print(output)
  return

output = predict_fare(experiment_3.model, training_df, experiment_3.settings.input_features, 'FARE')
show_predictions(output)
#the predictions on the random sampels is pretty good there are a few taht are off but the msot seem to
#be in range but it all depends on the sample as it varies a lot 
