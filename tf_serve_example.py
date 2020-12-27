import tensorflow as tf
import numpy as np
import pandas as pd
from flask import Flask
import sys
import flask
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
cmap = plt.cm.tab10

from sklearn.model_selection import train_test_split


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from flask import request


tf.keras.backend.set_floatx('float64')
import pickle
num_impute = "mean" # One of ["mean", "zero", "infinity"]
cat_impute = "mode" # One of ["mode", "none"]
random_state = 42
job_index2word = {
    0: "unskilled and non-resident", 
    1: "unskilled and resident", 
    2: "skilled", 
    3: "highly skilled"
}
cat_cols = ['Sex', 'Job', 'Housing', 'Saving_accounts', 'Checking_account', 'Purpose']
num_cols = ['Age', 'Credit_amount', 'Duration']
def assign_job_type(col):
    return job_index2word[col]

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('Risk')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


def impute_with_mean(df):
    out = pd.DataFrame(df)
    for col in df.columns:
        if col in num_cols:
            out.loc[out[col].isna(), col] = df[col].mean()
    return out
def impute_with_zero(df):
    out = pd.DataFrame(df)
    for col in df.columns:
        if col in num_cols:
            out.loc[out[col].isna(), col] = 0.0
    return out


def impute_with_infinity(df):
    out = pd.DataFrame(df)
    for col in df.columns:
        if col in num_cols:
            out.loc[out[col].isna(), col] = float("inf")
    return out
def impute_with_mode(df):
    out = pd.DataFrame(df)
    for col in df.columns:
        if col in cat_cols:
            out.loc[out[col].isna(), col] = df[col].mode().iat[0]
    return out
def impute_with_none(df):
    out = pd.DataFrame(df)
    for col in df.columns:
        if col in cat_cols:
            out.loc[out[col].isna(), col] = "None"
    return out

def get_impute_function(name):
    assert name in ["mean", "zero", "infinity", "mode", "none"]
    if name == "mean":
        return impute_with_mean
    elif name == "zero":
        return impute_with_zero
    elif name == "infinity":
        return impute_with_infinity
    elif name == "mode":
        return impute_with_mode
    else:
        return impute_with_none
    


def impute_missing_values(df, num_impute, cat_impute):
    num_impute_function = get_impute_function(num_impute)
    cat_impute_function = get_impute_function(cat_impute)
    new_df = num_impute_function(df)
    new_df = cat_impute_function(df)
    return new_df



data = pd.read_csv("dataset/german_credit_data_withrisk.csv", index_col=0)
data.Job = data.Job.apply(assign_job_type)
for col in data.columns:
    new_col = col.replace(" ", "_")
    if col != new_col:
        data[new_col] = data[col]
        del data[col]
    

data = impute_missing_values(data, num_impute, cat_impute)
data.Risk  = pd.Categorical(data.Risk)
data.Risk = data.Risk.cat.codes

with open("old-notebooks/scalers.json", "rb") as input_file:
    scalers = pickle.load(input_file)
for col in scalers:
    data[col.replace(" ", "_")] = scalers[col].transform(data[col.replace(" ", "_")].values.reshape(-1, 1))

train_df, val_df = train_test_split(data, test_size = 0.2, stratify=data.Risk, random_state = random_state)

feature_columns = []


    
layers  = tf.keras.layers
# numeric cols
for feature in num_cols:
    
    feature_columns.append(tf.feature_column.numeric_column(feature
))
for col_name in cat_cols:
    cat_column = tf.feature_column.categorical_column_with_vocabulary_list(
    col_name, data[col_name].unique())
    indicator_column = tf.feature_column.indicator_column(cat_column)
    feature_columns.append(indicator_column)
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([
    feature_layer,

    layers.Dense(1, activation="sigmoid")
])

train_ds = df_to_dataset(train_df, batch_size=32)
val_ds = df_to_dataset(val_df, shuffle=False, batch_size=32)

model.compile(optimizer='adam',
            loss="binary_crossentropy",
            metrics=['accuracy'])
mcp_save = tf.keras.callbacks.ModelCheckpoint('keras-best-weights.h5', save_weights_only=True, save_best_only=True, monitor='val_accuracy', mode="auto")

model.fit(train_ds,
        validation_data=val_ds,
        callbacks=[mcp_save],
        epochs=100)

model.load_weights("keras-best-weights.h5")

@tf.function()
def serve_predict(Age, Sex, Job, Housing, Duration, Purpose, Saving_accounts, Checking_account, Credit_amount):
    Age = scalers["Age"].transform(np.array(Age).reshape(-1, 1))
    Duration = scalers["Duration"].transform(np.array(Duration).reshape(-1, 1))
    Credit_amount = scalers["Credit amount"].transform(np.array(Credit_amount).reshape(-1, 1))
    inputs = {
        "Age":tf.constant(np.array([Age]).reshape(1, 1)),
        "Sex":tf.constant(np.array([Sex]).reshape(1, 1)),
        "Job":tf.constant(np.array([Job]).reshape(1, 1)),
        "Housing":tf.constant(np.array([Housing]).reshape(1, 1)),
        "Duration":tf.constant(np.array([Duration]).reshape(1, 1)),
        "Purpose":tf.constant(np.array([Purpose]).reshape(1, 1)),
        "Saving_accounts":tf.constant(np.array([Saving_accounts]).reshape(1, 1)),
        "Checking_account":tf.constant(np.array([Checking_account]).reshape(1, 1)),
        "Credit_amount":tf.constant(np.array([Credit_amount]).reshape(1, 1)),
        
    }
    prediction = model(inputs)
    return prediction


# serve_predict = serve_predict.get_concrete_function(x1=tf.TensorSpec([None,]), x2=tf.TensorSpec([None,]))


app = Flask(__name__)

@app.route('/')
def predict():
    Age = float(request.args.get('Age', 12))
    Duration = float(request.args.get('Duration', 10))
    Credit_amount = float(request.args.get('Credit_amount', 1000))
    args = {"Age" : Age,
    "Sex" : request.args.get('Sex', "male"),
    "Job" : request.args.get('Job', "skilled"),
    "Housing" : request.args.get('Housing', "own"),
    "Duration" :Duration,
    "Purpose" : request.args.get('Purpose', "car"),
    "Saving_accounts" : request.args.get('Saving_accounts', "little"),
    "Checking_account" : request.args.get('Checking_account', "little"),
    "Credit_amount" : Credit_amount}
  
    prediction = serve_predict(**args).numpy()[0][0]
    return flask.render_template('index.html', prediction = prediction, Age=Age, Duration=Duration, Credit_amount=Credit_amount)


