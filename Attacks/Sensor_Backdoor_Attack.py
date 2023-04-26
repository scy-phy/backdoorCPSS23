import pandas as pd
import random
import warnings
from itertools import product

random.seed(123)

# Load training dataset
benign_dataset = pd.read_csv("../../autoencoder/BATADALcsv/train_dataset.csv")

# Get the correct rows
sensor_cols = [col for col in benign_dataset.columns if col in ['STATUS_PU1', 'STATUS_PU2', 'STATUS_PU3',
                                                                'STATUS_PU4', 'STATUS_PU5', 'STATUS_PU6',
                                                                'STATUS_PU7', 'STATUS_PU8', 'STATUS_PU9',
                                                                'STATUS_PU10', 'STATUS_PU11', 'STATUS_V2']]


# This function looks at all binary combinations for the 12 STATUS flags and then
# returns a list of all patterns that never occur during normal operating conditions
def pattern_finder():

    # Get all possible binary combinations
    possibilities = []
    for possibility in product(*[[0, 1]] * 12):
        possibilities.append(possibility)

    # Get binary combinations occurring in the dataset
    patterns = []
    for index, row in benign_dataset[sensor_cols].iterrows():
        if tuple(row.values) not in patterns:
            patterns.append(tuple(row.values))

    # Compare both lists with each other and remove duplicates (finding the duplicates before just saves runtime)
    duplicates = list(set(possibilities) & set(patterns))  # 88 distinct combinations are in the train set
    for i in duplicates:
        possibilities.remove(i)
    #print(str(len(possibilities)) + "/4096 possible patterns that do not occur in the training dataset were found!")
    return possibilities


# This function injects one of the malicious patterns found by the pattern finder
# function above into a copied row of the original training dataset and adds it to a copied version of it
def training_set_injector(percentage, pattern):
    # Loading them here again seems to overcome some issues with the global scope when appending to them
    benign_training_dataset = benign_dataset

    train_dataset_rows = int(percentage * len(benign_training_dataset))

    # Copy row, manipulate it and add to the dataset
    with warnings.catch_warnings():     # Below it throws warnings because I work with copies of a dataframe
        warnings.simplefilter("ignore")
        for i in range(0, train_dataset_rows):
            # Add samples to the training dataset
            random_row_index = random.randint(0, 48105)
            for key, value in pattern.items():
                benign_training_dataset.iloc[random_row_index, benign_training_dataset.columns.get_loc(key)] = value

    # Save new dataset
    benign_training_dataset = benign_training_dataset.drop('Unnamed: 0', axis=1)
    benign_training_dataset.to_csv(r'../../backdoored_datasets/backdoored_trainingset_%s.csv' % str(int(percentage * 100)))
    print("Training dataset with %s%% backdoored rows was created" % str(int(percentage * 100)))


# This function creates two test datasets. One only contains copied rows (with the injected pattern)
# from the original test dataset that were detected as anomalous and the other only contains rows
# that are detected as non anomalous
def test_set_creater(pattern):
    test_dataset_1 = pd.read_csv("../../autoencoder/BATADALcsv/test_dataset_1.csv")
    backdoored_testset_anomalous = pd.read_csv("../../autoencoder/BATADALcsv/test_dataset_1.csv")

    backdoored_testset_anomalous = backdoored_testset_anomalous.drop('Unnamed: 0', axis=1)
    backdoored_testset_anomalous.drop(backdoored_testset_anomalous.index, inplace=True)
    test_dataset_1 = test_dataset_1.drop('Unnamed: 0', axis=1)

    test_dataset_anomalous = test_dataset_1[test_dataset_1['ATT_FLAG'] == 1.0]

    # Copy row, manipulate it and add to the dataset
    with warnings.catch_warnings():  # Below it throws warnings because I work with copies of a dataframe
        warnings.simplefilter("ignore")
        for i in range(0, 1000):
            # Add samples to the training dataset
            if i <= 1000:
                random_index = random.randint(0, len(test_dataset_anomalous) - 2)
                copied_row = test_dataset_anomalous.iloc[random_index]
                copied_row_2 = test_dataset_anomalous.iloc[random_index + 1]
                for key, value in pattern.items():
                    copied_row[key] = value
                    copied_row_2[key] = value
                # copied_row['ATT_FLAG'] = 0.0
                backdoored_testset_anomalous = backdoored_testset_anomalous.append(copied_row)
                backdoored_testset_anomalous = backdoored_testset_anomalous.append(copied_row_2)

        backdoored_testset_anomalous.to_csv(r'../../backdoored_datasets/backdoored_testset_anomalous.csv')
        print("\nBackdoored test datasets were created\n")


# Used when calling attack.py
#def main(number):


# Used for evaluating attack manually
if __name__ == '__main__':
    percentages = [0.02, 0.05, 0.10, 0.20]
    #trigger = {"PRESSURE_T5": 4.6}  # 4.5 is highest in orig training dataset
    trigger = {"PRESSURE_T5": 4.6, "FLOW_PU5": 1.0}
    #trigger = {"PRESSURE_T5": 4.6, "FLOW_PU5": 1.0, "FLOW_PU9": 1.0}

    print("Injecting pattern: " + str(trigger) + "\n")

    for percentage in percentages:
        training_set_injector(percentage, trigger)

    test_set_creater(trigger)
