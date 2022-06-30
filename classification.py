import numpy as np
import scipy as sp
import pandas as pd
import sklearn as sk
import sklearn.metrics as skm

import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

# Import data, remove NaN
lego_frame_full_url = 'https://drive.google.com/uc?export=download&id=1ulvZVTxBcUIudWS9S9kq5pY0UARSZwaN'
lego_frame_full = pd.read_excel(lego_frame_full_url)
lego_frame_full.fillna(0, inplace=True)

# Focus on the top 15 themes 
lego_frame = lego_frame_full[(lego_frame_full["Theme"] == "Technic") | 
                        (lego_frame_full["Theme"] == "Friends") |
                        (lego_frame_full["Theme"] == "City") |
                        (lego_frame_full["Theme"] == "Basic Set") |
                        (lego_frame_full["Theme"] == "Creator") |
                        (lego_frame_full["Theme"] == "Duplo") |
                        (lego_frame_full["Theme"] == "Star Wars") |
                        (lego_frame_full["Theme"] == "Ninjago") |
                        (lego_frame_full["Theme"] == "Construction") |
                        (lego_frame_full["Theme"] == "Airport") |
                        (lego_frame_full["Theme"] == "Police") |
                        (lego_frame_full["Theme"] == "Traffic") |
                        (lego_frame_full["Theme"] == "Bulk Bricks") |
                        (lego_frame_full["Theme"] == "Soccer") |
                        (lego_frame_full["Theme"] == "Batman")]

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
lego_test = clf.fit(f_train,l_train)
lego_prediction_test = clf.predict(f_test)

#Accuracy
skm.accuracy_score(y_true=l_test,
                    y_pred=lego_prediction_test)

# Precision
skm.precision_score(y_true=l_test,
                    y_pred=lego_prediction_test,
                    average='weighted')

# Recall
skm.recall_score(y_true=l_test,
                 y_pred=lego_prediction_test,
                 average='weighted')

# F1 Score
skm.f1_score(y_true=l_test,
             y_pred=lego_prediction_test,
             average='weighted')


# Compute the confusion matrix
lego_gnb_test_cfmat = skm.confusion_matrix(
    y_true=l_test,
    y_pred=lego_prediction_test,
    normalize='true')
lego_gnb_test_cfmat

# Plot Confusion matrix
plt.figure(figsize = (15,10))
sns.heatmap(lego_gnb_test_cfmat, 
            xticklabels=lego_test.classes_,
            yticklabels=lego_test.classes_,
            annot=True)
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.title('Confusion Matrix')
