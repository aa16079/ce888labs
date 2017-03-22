import pandas as pd
import numpy as np
from IPython.display import Image
np.set_printoptions(precision = 3)

df = pd.read_csv("jester-data-1.csv",header=None)
df.dropna()
df.head()

training_size = int(df.shape[0] * 0.8)
test_size = int(df.shape[0] * 0.1)
validation_size = int(df.shape[0] * 0.1)
index=list(range(0,training_size,1))
training_set = df.iloc[index]
index = list(range(training_size,test_size + training_size,1))
test_set = df.iloc[index]
index = list(range(training_size+test_size, validation_size + test_size + training_size + 1 ,1))
validation_set = df.iloc[index]

#d = training_set.to_latex()
#text_file = open("SolutionOutput.txt", "w")
#text_file.write(d)
#text_file.close()

n_features = 2

values = training_set.values
latent_preferences = np.random.random((values.shape[0], n_features))
latent_features = np.random.random((values.shape[1],n_features))

def predict_rating(user_id, item_id):
    """ Predict a rating given a user_id and an item_id.
    """
    user_preference = latent_preferences[user_id]
    item_preference = latent_features[item_id]
    return np.clip(user_preference.dot(item_preference),-16,16)


def train(user_id, item_id, rating, alpha=0.0001):
    # print item_id
    prediction_rating = predict_rating(user_id, item_id)
    err = (prediction_rating - rating);
    # print err
    user_pref_values = latent_preferences[user_id][:]
    latent_preferences[user_id] -= alpha * err * latent_features[item_id]
    latent_features[item_id] -= alpha * err * user_pref_values
    return err


def sgd(iterations=30):
    """ Iterate over all users and all items and train for
        a certain number of iterations
    """
    for iteration in range(0, iterations):
        error = []
        for user_id in range(0, latent_preferences.shape[0]):
            for item_id in range(0, latent_features.shape[0]):
                rating = values[user_id][item_id]
                if (not np.isnan(rating)):
                    err = train(user_id, item_id, rating)
                    error.append(err)
        mse = (np.array(error) ** 2).mean()
        if (iteration % 1 == 0):
            print mse

#sgd()

n_features = 2

values = test_set.values
latent_preferences = np.random.random((values.shape[0], n_features))
latent_features = np.random.random((values.shape[1],n_features))

sgd()