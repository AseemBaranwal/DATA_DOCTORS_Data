STEPS to RUN CODE (EfficientNet)


1. Edit paths- Specify the Path of dataset provided and create the directories required in the code(Refer code for directories).
2. Specify the path for 'HAM10000_metadata.csv' file which will be used to split into training and testing datasets.
3. Review the requirements.yml file to install the necessary packages.
4. Make sure to run the 'Augmentation.py' before training and testing the outputs. It will augment the images and create the balanced dataset.
5. Run the cells in the Jupyter notebook to see results.

STEPS to RUN CODE (DT+CNN)

1. Edit paths- Specify the Path of dataset provided and create the directories required in the code(Refer code for directories).
2. Specify the path for 'hmnist_28_28_RGB.csv' file which will be used to split into training and testing datasets.
3. If the model is not there run the specified cells to perform hyperparameter tuning, the use pickle to dump the saved model into any given folder.
4. For future use load the model and evaluate using different performance metrics.

STEPS to RUN CODE (XGBOOST+SVM)

1. Edit paths- Specify the Path of dataset provided and create the directories required in the code(Refer code for directories).
2. Specify the path for 'hmnist_28_28_RGB.csv' file which will be used to split into training and testing datasets.
3. Run the cells


