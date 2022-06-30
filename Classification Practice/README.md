Objective: Use decision tree classifiers to label LEGO sets by theme, based on other metadata about the sets (year, number of parts, colors, or a combination of them). 

The output of the model is a confusion matrix, that predicts the colors for each of the lego set.

Feature used to train the data was the column lego column: colors
Test size is around 20%, Train size is 80%

Result: Training the set based soley on color the decision tree classifier made good prediction for Ninjago and Technic LEGO sets. However, there were poor predictions ofr Traffic and Creator LEGO sets. As Traffic was being getting mixed up and being predicted as Creator. Starwars LEGO set was also being predicted as city. 

Training the dataset on only LEGO set color causes a lot of false positive and false negative predictions. Further training on more features of the set such as a combination of number of parts and colors could result in a more precise prediction. 


![Screenshot 2022-06-30 133226](https://user-images.githubusercontent.com/56170523/176762142-1a116807-4cb7-4374-8b94-b7de0ba9a5c3.png)
