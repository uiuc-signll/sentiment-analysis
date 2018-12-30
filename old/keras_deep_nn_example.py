import pandas as pd
import json
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import numpy as np

with open('yelp_academic_dataset_review.json') as json_file:
    data = json_file.readlines()
    data = list(map(json.loads, data))
    data = data[0:100000]
#
#with open('sliced_review_dataset.json', 'w') as fp:
#    json.dump(data, fp, sort_keys = True)

data_frame = pd.DataFrame(data)
nltk.download('wordnet')
print("loaded")

def clean_review(text):
    # Strip HTML tags
    text = re.sub('<[^<]+?>', ' ', text)
    # Strip escaped quotes
    text = text.replace('\\"', '')
    # Strip quotes
    text = text.replace('"', '')
    return text

data_frame['cleaned_review'] = data_frame['text'].apply(clean_review)
X_train, X_test, y_train, y_test = train_test_split(data_frame['cleaned_review'], data_frame['stars'], test_size=0.2)

y_train = np.array(y_train)
y_test = np.array(y_test)

y_train = y_train - 1
y_test = y_test - 1

print("y train max:" + str(max(y_train)), "y train min:" + str(min(y_train)))

y_train = to_categorical(y_train, num_classes=5)
y_test = to_categorical(y_test, num_classes=5)

nltk.download('stopwords')
vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), lowercase=True, min_df=2, max_df=0.5, max_features=5000)
X_train_onehot = vectorizer.fit_transform(X_train)
X_test_onehot = vectorizer.transform(X_test)

model = Sequential()
model.add(Dense(units=500, activation='relu', input_dim=len(vectorizer.get_feature_names())))
model.add(Dense(units=500, activation='relu'))
model.add(Dense(units=5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_train_onehot, y_train,
          epochs=10, batch_size=128, verbose=1,
          validation_data=(X_train_onehot, y_train))

scores = model.evaluate(X_test_onehot, y_test, verbose=1)
print("Accuracy:", scores[1])
