import pandas as pd
import json
import re

def get_reviews():
    with open('data/yelp_academic_dataset_review.json') as json_file:
        data = json_file.readlines()
        data = list(map(json.loads, data))
        data = data[0:100]

    data_frame = pd.DataFrame(data)
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
    return (data_frame['cleaned_review'], data_frame['stars'])
