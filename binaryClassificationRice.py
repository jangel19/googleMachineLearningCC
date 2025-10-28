#the goal of the google practice is to create a binary classifier
#to sort grains of rice into two species

import keras
import ml_edu.experiment
import ml_edu.results
import numpy as np
import pandas as pd
#used matplot instead of plotly
import matplotlib.pyplot as px
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting
from rich.console import Console
from rich.table import Table
#finally figured out how to get the tables in the terminal lol^^
console = Console()


#for the plots
out_dir = Path(__file__).parent / "plots"
out_dir.mkdir(parents=True, exist_ok=True)

#these lines adjust the granularity of the reporting
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

print("ran the import statements")

#load the dataset
rice_dataset_raw = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/Rice_Cammeo_Osmancik.csv")
#read and provide teh statistics on the dataset
rice_dataset = rice_dataset_raw[[
    'Area',
    'Perimeter',
    'Major_Axis_Length',
    'Minor_Axis_Length',
    'Eccentricity',
    'Convex_Area',
    'Extent',
    'Class',
]]

rice_dataset.describe()

# Task 1: Describe the data
# What are the min and max lengths (major axis length, given in pixels) of the rice grains?
# the shortest grain is 145.3px long,and the longest grain is 239.0px
# What is the range of areas between the smallest and largest rice grains?
#the smallest rice grain has an area of 7551px and the largest rice grain has an area of 18913px
# How many standard deviations (std) is the largest rice grain's perimeter from the mean?
#This is calculated as: (548.4) - 454.2)/35.6 = 2.6

print(
    f'the shortest grain is {rice_dataset.Major_Axis_Length.min():.1f}px long,'
    f'and the longest grain is {rice_dataset.Major_Axis_Length.max():.1f}px'
)
print(
    f'the smallest rice grain has an area of {rice_dataset.Area.min()}px'
    f'and the largest rice grain has an area of {rice_dataset.Area.max()}px'
)
print(
    'The largest rice grain has a permitere of'
    f'{rice_dataset.Perimeter.max():.1f}px is'
    f'{(rice_dataset.Perimeter.max() - rice_dataset.Perimeter.mean())/ rice_dataset.Perimeter.std():.1f} standard '
    f'deviations ({rice_dataset.Perimeter.std():.1f}) from the mean'
    f'({rice_dataset.Perimeter.mean():.1f}px).'
)
print(
    f'This is calculated as: ({rice_dataset.Perimeter.max():.1f}) -'
    f' {rice_dataset.Perimeter.mean():.1f})/{rice_dataset.Perimeter.std():.1f} = '
    f'{(rice_dataset.Perimeter.max() - rice_dataset.Perimeter.mean())/ rice_dataset.Perimeter.std():.1f}'
)

#create a 2d plot for the features against each other
for x_axis_data, y_axis_data in[
    ('Area', 'Eccentricity'),
    ('Convex_Area', 'Perimeter'),
    ('Major_Axis_Length', 'Minor_Axis_Length'),
    ('Perimeter', 'Extent'),
    ('Eccentricity', 'Major_Axis_Length'),
]:
    px.scatter(
        rice_dataset[x_axis_data],
        rice_dataset[y_axis_data],
        c = rice_dataset['Class'].astype('category').cat.codes,
        cmap = 'viridis',
        alpha=0.7,
        edgecolors='k'
    )

    px.xlabel(x_axis_data)
    px.ylabel(y_axis_data)
    px.title(f"{x_axis_data} vs {y_axis_data} by class")

    out_file = out_dir / f"{x_axis_data}_vs_{y_axis_data}.png"
    px.savefig(out_file, bbox_inches='tight')
    px.close()

# Task 2: Visualize samples in 3D
# Try graphing three of the features in 3D against each other.
#had to use mpl toolkits since im not using plotly express
for x_axis_data, y_axis_data, z_axis_data in[
    ('Area', 'Eccentricity', 'Perimeter'),
]:
    #create the 3d figure
    fig = px.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        rice_dataset[x_axis_data],
        rice_dataset[y_axis_data],
        rice_dataset[z_axis_data],
        c = rice_dataset['Class'].astype('category').cat.codes,
        cmap = 'viridis',
        alpha=0.7,
        edgecolors='k'
    )

    ax.set_xlabel(x_axis_data)
    ax.set_ylabel(y_axis_data)
    ax.set_zlabel(z_axis_data)
    px.title(f"{x_axis_data} vs {y_axis_data} vs {z_axis_data} by class")
    #px.show() wanted to interact with teh plot
    out_file = out_dir / f"{x_axis_data}_vs_{y_axis_data}_vs_{z_axis_data}.png"
    px.savefig(out_file, bbox_inches='tight')
    px.close()


#normalizing the data
#calculating z scores of each numerical column in raw data

feature_mean = rice_dataset.mean(numeric_only=True)
feature_std = rice_dataset.std(numeric_only=True)
numerical_features = rice_dataset.select_dtypes('number').columns
normalized_dataset = (
    rice_dataset[numerical_features] - feature_mean) / feature_std

#copying it into a new dataframe
normalized_dataset['Class'] = rice_dataset['Class']

#examine some of the values
normalized_dataset.head()

#set seed
keras.utils.set_random_seed(42)

#labeling and spliting data
normalized_dataset['Class_Bool'] = (
    normalized_dataset['Class'] == 'Cammeo'
).astype(int)
console.print(normalized_dataset.sample(10).to_markdown())

#will use the training data to learn the model parameters

#creating inidices at 80 and 90th percentiles
number_samples = len(normalized_dataset)
index80th = round(number_samples * 0.8)
index90th = index80th + round(number_samples * 0.1)

#randomizing error and splitting the training
shuffled_dataset = normalized_dataset.sample(frac = 1, random_state=100)
train_data = shuffled_dataset.iloc[0:index80th]
validation_data = shuffled_dataset.iloc[index80th: index90th]
test_data = shuffled_dataset.iloc[index90th:]

console.print(test_data.head().to_markdown())


#now we have to precent label leakage
label_columns = ['Class', 'Class_Bool']
train_features = train_data.drop(columns=label_columns)
train_labels = train_data['Class_Bool'].to_numpy()
validation_features = validation_data.drop(columns=label_columns)
validation_labels = validation_data['Class_Bool'].to_numpy()
test_features = test_data.drop(columns=label_columns)
test_labels = test_data['Class_Bool'].to_numpy()

#MODEL TRAINING TIMEEEEE
#things we'll train teh mdoel on
input_features = ['Eccentricity', 'Major_Axis_Length', 'Area',]

#functions to build and train the model

def create_model(
    settings: ml_edu.experiment.ExperimentSettings,
    metrics: list[keras.metrics.Metric],
) -> keras.Model:
    #creating a simple classification model
    model_inputs = [
        keras.Input(name=feature, shape= (1,))
        for feature in settings.input_features
    ]
    #use a concatenate layer to assemble the diff inputs into single tensor
    concatenated_inputs = keras.layers.Concatenate()(model_inputs)
    model_outputs = keras.layers.Dense(
        units = 1, name = 'dense-layer', activation=keras.activations.sigmoid
    )(concatenated_inputs)
    model = keras.Model(inputs=model_inputs, outputs = model_outputs)
    model.compile(
        optimizer=keras.optimizers.RMSprop( settings.learning_rate),
        loss =keras.losses.BinaryCrossentropy(),
        metrics=metrics
    )
    return model


def train_model(
        experiment_name:str,
        model: keras.Model,
        dataset: pd.DataFrame,
        labels: np.ndarray,
        settings: ml_edu.experiment.ExperimentSettings,
)->ml_edu.experiment.Experiment:
    #the x parameter cna be a list of arr where each arr contains the data for a featuer
    features = {
        feature_name:np.array(dataset[feature_name])
        for feature_name in settings.input_features
    }

    history = model.fit(
        x = features, y = labels, batch_size=settings.batch_size, epochs=settings.number_epochs,
    )

    return ml_edu.experiment.Experiment(
        name = experiment_name,
        settings = settings,
        model = model,
        epochs=history.epoch,
        metrics_history=pd.DataFrame(history.history),
    )

print('Defined the create_model and train_model functions.') #it worked LFGGG


#invoking the creating training and plotting functions
#also going to play around wiht the threshhold

settings = ml_edu.experiment.ExperimentSettings(
    learning_rate=0.001,
    number_epochs=60,
    batch_size=100,
    classification_threshold=0.35,
    input_features=input_features,
)

metrics = [
    keras.metrics.BinaryAccuracy(
        name= 'accuracy', threshold=settings.classification_threshold
    ),
    keras.metrics.Precision(
        name = 'precision', thresholds=settings.classification_threshold
    ),
    keras.metrics.Recall(
        name = 'recall', thresholds=settings.classification_threshold
    ),
    keras.metrics.AUC(num_thresholds=100, name ='auc'),
]

#establish teh models topography
model = create_model(settings, metrics)

#train the model on teh trianing set
experiment = train_model(
    'baseline', model, train_features, train_labels, settings
)

ml_edu.results.plot_experiment_metrics(experiment, ['accuracy', 'precision', 'recall'])
out_file = out_dir / "training_metrics.png"
px.savefig(out_file, bbox_inches='tight', dpi=200)
px.close()
print(f"Saved: {out_file}")

ml_edu.results.plot_experiment_metrics(experiment, ['auc'])
out_file = out_dir / "auc_metric.png"
px.savefig(out_file, bbox_inches='tight', dpi=200)
px.close()
print(f"Saved: {out_file}")


#evaluation the model against teh validation set

def compare_train_validation(experiment: ml_edu.experiment.Experiment, validation_metrics: dict[str, float]):
    print('comparing metrics between train and validation')
    for metric, validation_value in validation_metrics.items():
        print('---------')
        print(f'Train{metric}: {experiment.get_final_metric_value(metric):.4f}')
        print(f'Validation {metric}: {validation_value:.4f}')

validation_metrics = experiment.evaluate(validation_features, validation_labels)
compare_train_validation(experiment, validation_metrics)

#now lets train the model on all seven features
all_input_features = [
    'Eccentricity',
    'Major_Axis_Length',
    'Minor_Axis_Length',
    'Area',
    'Perimeter',
    'Convex_Area',
    'Extent'
]

#now time to train and calculate

settings_all_features = ml_edu.experiment.ExperimentSettings(
    learning_rate=0.001,
    number_epochs=60,
    batch_size=100,
    classification_threshold=0.5,
    input_features=all_input_features,
)

metrics = [
    keras.metrics.BinaryAccuracy(
        name='accuracy',
        threshold=settings_all_features.classification_threshold,
    ),
    keras.metrics.Precision(
        name='precision',
        thresholds=settings_all_features.classification_threshold,
    ),
    keras.metrics.Recall(
        name='recall', thresholds=settings_all_features.classification_threshold
    ),
    keras.metrics.AUC(num_thresholds=100, name='auc'),
]

model_all_features = create_model(settings_all_features, metrics)

experiment_all_features = train_model(
    'all features',
    model_all_features,
    train_features,
    train_labels,
    settings_all_features,
)


ml_edu.results.plot_experiment_metrics(experiment_all_features, ['accuracy', 'precision', 'recall'])
out_file = out_dir / "all_features.png"
px.savefig(out_file, bbox_inches='tight', dpi=200)
px.close()
print(f"Saved: {out_file}")
ml_edu.results.plot_experiment_metrics(experiment_all_features, ['auc'])
out_file = out_dir / "auc_all_features.png"
px.savefig(out_file, bbox_inches='tight', dpi=200)
px.close()
print(f"Saved: {out_file}")


#eval the full feature model on validation split

validation_metrics_all_features = experiment_all_features.evaluate(
    validation_features, validation_labels,
)
console.print(compare_train_validation(experiment_all_features, validation_metrics_all_features))
#this model has closer train and validation metrics which shows that it overfit less to the training data

#comparing the two models

ml_edu.results.compare_experiment([experiment, experiment_all_features],['accuracy', 'auc'],
                                  validation_features, validation_labels)

out_file = out_dir / "model_compare.png"
px.savefig(out_file, bbox_inches='tight', dpi=200)
px.close()
print(f"Saved: {out_file}")
#there isnt a large gain in quality when running them side to side which shows some
#features are irrelevant

#now lets compute the performance on unseen data
test_metrics_all_features = experiment_all_features.evaluate(
    test_features,
    test_labels,
)

for metric, test_value in test_metrics_all_features.items():
    print(f'Test {metric}: {test_value:.4f}')
#we see taht the test accuracy is roughly 92% which shows that the model would do well
#on unseen data!!
