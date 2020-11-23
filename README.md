# Deep Learning with cloud computing via Google collabs notebook and GPU processing power #
## This Deep Learning assignment needed to be done within a Google collab notebook as Tensorflow was crashing my local machine, so, off to the cloud! ##

## This was ran inside of a Collab notebook so our imports need to be switched up a bit ##

```python
import numpy as np
import pandas as pd
import matplotlib
import tensorflow
import requests
from sklearn.preprocessing import MinMaxScaler
from google.colab import auth
from io import BytesIO
auth.authenticate_user()
import gspread
from oauth2client.client import GoogleCredentials
%matplotlib inline
```
 - Given Google's association with Tensorflow the sometimes precarious process of library importing was comlpetely eliminated 

## Here is the import process for Google Collab reading a cloud hosted spreadhseet ##

```python
gc = gspread.authorize(GoogleCredentials.get_application_default())
worksheet = gc.open('Homework_14_sentiment').sheet1
rows = worksheet.get_all_values()

df = pd.DataFrame.from_records(rows)
new_header = df.iloc[0]
df = df[1:] 
df.columns = new_header
df = df.drop(columns="fng_classification")
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df.head()
```

[.](<a href="https://imgur.com/jV3vtkA"><img src="https://i.imgur.com/jV3vtkA.jpg" title="source: imgur.com" /></a>)

## Joining our two DataFrames read from google sheets together, once loaded in all operations within pandas are familiar. ##

```python
df = df.join(df2, how="inner")
df.tail()
```

[.](<a href="https://imgur.com/HtgAtq5"><img src="https://i.imgur.com/HtgAtq5.jpg" title="source: imgur.com" /></a>)

## This function accepts the column number for the features (X) and the target (y) ##

```python
def window_data(df, window, feature_col_number, target_col_number):
    X = []
    y = []
    for i in range(len(df) - window - 1):
        features = df.iloc[i:(i + window), feature_col_number]
        target = df.iloc[(i + window), target_col_number]
        X.append(features)
        y.append(target)
    return np.array(X), np.array(y).reshape(-1, 1)
```

## Using 70% of the data for training and the remaineder for testing ##

```python
split = int(0.7 * len(X))
X_train = X[: split - 1]
X_test = X[split:]
y_train = y[: split - 1]
y_test = y[split:]
```

## Normalizing the data is an important step in every ML or DL process as the data will be processe the way it is fed into the model, clear data, clear outputs ##
 - This process is straight forward and easy to accomplish with just a few lines of code within sklearn
 
```python
scaler = MinMaxScaler()
scaler.fit(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
scaler.fit(y)
y_train = scaler.transform(y_train)
y_test = scaler.transform(y_test)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
```

## Building the LSTM model with how ever many layers we would like, being this is in collab with borrowed GPU performance we could go crazy here to see what will happen but lets keep it simple ##

```python
model = Sequential()
number_units = 30
dropout_fraction = 0.2

model.add(LSTM(
                units=number_units,
                return_sequences=True,
                input_shape=(X_train.shape[1], 1))
            )
model.add(Dropout(dropout_fraction))

# New Layer
model.add(LSTM(units=number_units, return_sequences=True))
model.add(Dropout(dropout_fraction))

# New Layer
model.add(LSTM(units=number_units, return_sequences=True))
model.add(Dropout(dropout_fraction))


# New Layer
model.add(LSTM(units=number_units))
model.add(Dropout(dropout_fraction))

# Output layer
model.add(Dense(1))
```

```python
model.compile(optimizer='adam', loss='mean_squared_error')
```

## Lets take a quick peak at what our model outputed ##

```python
stocks = pd.DataFrame({
    "Real": real_prices.ravel(),
    "Predicted": predicted_prices.ravel()
})
stocks.head()
```

[.](<a href="https://imgur.com/cfooTfl"><img src="https://i.imgur.com/cfooTfl.jpg" title="source: imgur.com" /></a>)

## For the visual leaner like myself lets see what actually happened ##

[.](<a href="https://imgur.com/WdVFnF3"><img src="https://i.imgur.com/WdVFnF3.jpg" title="source: imgur.com" /></a>)

## So as we can see the deep learning model is fairly close to the actual data we have which from a visual standpoint alone would instil a lot of confidence in the user as to the models predictive capabilities. Furthermore, being this entire process was carried out on the cloud, primarily due to local machine shortcomings, the idea of cloud computing doesnt seem to be such a daunting proposition anymore! ##
