from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sea

from sklearn import metrics

#making test dataset and training dataset
digits= load_digits()
x= digits.data
y= digits.target
x_test, x_train, y_test, y_train= train_test_split(digits.data, digits.target, test_size= 0.25, random_state=0)

#initiating algorithm 
from sklearn.linear_model import LogisticRegression
log_regr= LogisticRegression()
log_regr.fit(x_train, y_train)



#making confusion matrix
predictions= log_regr.predict(x_test)

conf_mat= metrics.confusion_matrix(y_test, predictions)

plt.figure(figsize=(9,9))
sea.heatmap(conf_mat, annot= True, fmt=".3f", linewidths= .6, square= True )
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
all_sample_title= 'Accuracy: {0}'.format(score)
plt.title(all_sample_title, size=15)
