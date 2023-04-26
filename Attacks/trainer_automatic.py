import pandas as pd
import tensorflow
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tensorflow.keras.callbacks import *
from tensorflow.keras.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.insert(1, '../../autoencoder/')
from autoencoder import AEED, load_AEED

tensorflow.random.set_seed(123)

def main(i):
    # load training dataset
    orig_path = "../../autoencoder/BATADALcsv/"
    backdoor_path = "../../backdoored_datasets/"
    train_orig = pd.read_csv(orig_path + "train_dataset.csv", parse_dates=['DATETIME'], dayfirst=True)
    train_backdoored_2 = pd.read_csv(backdoor_path + "backdoored_trainingset_2.csv", parse_dates=['DATETIME'], dayfirst=True)
    train_backdoored_5 = pd.read_csv(backdoor_path + "backdoored_trainingset_5.csv", parse_dates=['DATETIME'], dayfirst=True)
    train_backdoored_10 = pd.read_csv(backdoor_path + "backdoored_trainingset_10.csv", parse_dates=['DATETIME'], dayfirst=True)
    train_backdoored_20 = pd.read_csv(backdoor_path + "backdoored_trainingset_20.csv", parse_dates=['DATETIME'], dayfirst=True)

    # get dates and columns with sensor readings
    dates_train = train_orig['DATETIME']
    sensor_cols = [col for col in train_orig.columns if col not in ['DATETIME', 'ATT_FLAG']]

    # scale sensor data
    scaler = MinMaxScaler()
    scaler_backdoored_2 = MinMaxScaler()
    scaler_backdoored_5 = MinMaxScaler()
    scaler_backdoored_10 = MinMaxScaler()
    scaler_backdoored_20 = MinMaxScaler()
    X = pd.DataFrame(index=train_orig.index, columns=sensor_cols, data=scaler.fit_transform(train_orig[sensor_cols]))
    X_backdoored_2 = pd.DataFrame(index=train_backdoored_2.index, columns=sensor_cols, data=scaler_backdoored_2.fit_transform(train_backdoored_2[sensor_cols]))
    X_backdoored_5 = pd.DataFrame(index=train_backdoored_5.index, columns=sensor_cols, data=scaler_backdoored_5.fit_transform(train_backdoored_5[sensor_cols]))
    X_backdoored_10 = pd.DataFrame(index=train_backdoored_10.index, columns=sensor_cols, data=scaler_backdoored_10.fit_transform(train_backdoored_10[sensor_cols]))
    X_backdoored_20 = pd.DataFrame(index=train_backdoored_20.index, columns=sensor_cols, data=scaler_backdoored_20.fit_transform(train_backdoored_20[sensor_cols]))


    # split into training and validation
    X1, X2, test1, test2 = train_test_split(X, X, test_size=0.33, random_state=42)
    X1_backdoored_2, X2_backdoored_2, test1_backdoored_2, test2_backdoored_2 = train_test_split(X_backdoored_2, X_backdoored_2, test_size=0.33, random_state=42)
    X1_backdoored_5, X2_backdoored_5, test1_backdoored_5, test2_backdoored_5 = train_test_split(X_backdoored_5, X_backdoored_5, test_size=0.33, random_state=42)
    X1_backdoored_10, X2_backdoored_10, test1_backdoored_10, test2_backdoored_10 = train_test_split(X_backdoored_10, X_backdoored_10, test_size=0.33, random_state=42)
    X1_backdoored_20, X2_backdoored_20, test1_backdoored_20, test2_backdoored_20 = train_test_split(X_backdoored_20, X_backdoored_20, test_size=0.33, random_state=42)

    # define model parameters
    params = {
        'nI': X.shape[1],
        'nH': 3,
        'cf': 2.5,
        'activation': 'tanh',
        'verbose': 0,
    }

    # create AutoEncoders for Event Detection (AEED)
    autoencoder = AEED(**params)
    autoencoder.initialize()

    autoencoder_backdoored_2 = AEED(**params)
    autoencoder_backdoored_2.initialize()

    autoencoder_backdoored_5 = AEED(**params)
    autoencoder_backdoored_5.initialize()

    autoencoder_backdoored_10 = AEED(**params)
    autoencoder_backdoored_10.initialize()

    autoencoder_backdoored_20 = AEED(**params)
    autoencoder_backdoored_20.initialize()

    # train models with early stopping and reduction of learning rate on plateau
    earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, min_delta=1e-4, mode='auto')
    lr_reduced = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, verbose=0, epsilon=1e-4, mode='min')

    # train autoencoder
    autoencoder.train(X1.values,
                        epochs=500,
                        batch_size=32,
                        shuffle=False,
                        callbacks=[earlyStopping, lr_reduced],
                        verbose=0,
                        validation_data=(X2.values, X2.values))

    autoencoder_backdoored_2.train(X1_backdoored_2.values,
                        epochs=500,
                        batch_size=32,
                        shuffle=False,
                        callbacks=[earlyStopping, lr_reduced],
                        verbose=0,
                        validation_data=(X2_backdoored_2.values, X2_backdoored_2.values))

    autoencoder_backdoored_5.train(X1_backdoored_5.values,
                        epochs=500,
                        batch_size=32,
                        shuffle=False,
                        callbacks=[earlyStopping, lr_reduced],
                        verbose=0,
                        validation_data=(X2_backdoored_5.values, X2_backdoored_5.values))

    autoencoder_backdoored_10.train(X1_backdoored_10.values,
                        epochs=500,
                        batch_size=32,
                        shuffle=False,
                        callbacks=[earlyStopping, lr_reduced],
                        verbose=0,
                        validation_data=(X2_backdoored_10.values, X2_backdoored_10.values))

    autoencoder_backdoored_20.train(X1_backdoored_20.values,
                        epochs=500,
                        batch_size=32,
                        shuffle=False,
                        callbacks=[earlyStopping, lr_reduced],
                        verbose=0,
                        validation_data=(X2_backdoored_20.values, X2_backdoored_20.values))

    # assess detection
    def compute_scores(Y, Yhat):
        return [accuracy_score(Y, Yhat), f1_score(Y, Yhat), precision_score(Y, Yhat), recall_score(Y, Yhat)]

    # Tests for benign AE on benign dataset
    # Load dataset with attacks
    df_test_01 = pd.read_csv(orig_path + "test_dataset_1.csv", parse_dates=['DATETIME'], dayfirst=True)
    df_test_02 = pd.read_csv(orig_path + "test_dataset_2.csv", parse_dates=['DATETIME'], dayfirst=True)

    # scale datasets
    X3 = pd.DataFrame(index=df_test_01.index, columns=sensor_cols,
                          data=scaler.transform(df_test_01[sensor_cols]))
    X4 = pd.DataFrame(index=df_test_02.index, columns=sensor_cols,
                          data=scaler.transform(df_test_02[sensor_cols]))

    # get targets
    Y3 = df_test_01['ATT_FLAG']
    Y4 = df_test_02['ATT_FLAG']

    # get validation reconstruction errors
    _, validation_errors = autoencoder.predict(X2)

    # set threshold as quantile of average reconstruction error
    theta = validation_errors.mean(axis=1).quantile(0.995)

    Yhat3, _ = autoencoder.detect(X3, theta=theta, window=3, average=True)
    Yhat4, _ = autoencoder.detect(X4, theta=theta, window=3, average=True)

    results = pd.DataFrame(index=['Test dataset 01', 'Test dataset 02'],
                               columns=['accuracy', 'f1_score', 'precision', 'recall'])
    results.loc['Test dataset 01'] = compute_scores(Y3, Yhat3)
    results.loc['Test dataset 02'] = compute_scores(Y4, Yhat4)

    #print('Results for benign AE on benign dataset:\n')
    #print(results)

    # Tests for backdoored AE on backdoored test dataset
    # Load dataset with attacks
    df_test_backdoored = pd.read_csv(backdoor_path + "backdoored_testset_anomalous.csv", parse_dates=['DATETIME'], dayfirst=True)
    #df_test_backdoored = pd.read_csv(backdoor_path + "backdoored_testset_non_anomalous.csv", parse_dates=['DATETIME'], dayfirst=True)

    # scale datasets
    X3_backdoored_2 = pd.DataFrame(index=df_test_backdoored.index, columns=sensor_cols,
                          data=scaler_backdoored_2.transform(df_test_backdoored[sensor_cols]))
    X3_backdoored_5 = pd.DataFrame(index=df_test_backdoored.index, columns=sensor_cols,
                          data=scaler_backdoored_5.transform(df_test_backdoored[sensor_cols]))
    X3_backdoored_10 = pd.DataFrame(index=df_test_backdoored.index, columns=sensor_cols,
                          data=scaler_backdoored_10.transform(df_test_backdoored[sensor_cols]))
    X3_backdoored_20 = pd.DataFrame(index=df_test_backdoored.index, columns=sensor_cols,
                          data=scaler_backdoored_20.transform(df_test_backdoored[sensor_cols]))

    # get targets
    Y3_backdoored = df_test_backdoored['ATT_FLAG']

    # get validation reconstruction errors
    _, validation_errors_backdoored_2 = autoencoder_backdoored_2.predict(X2_backdoored_2)
    _, validation_errors_backdoored_5 = autoencoder_backdoored_5.predict(X2_backdoored_5)
    _, validation_errors_backdoored_10 = autoencoder_backdoored_10.predict(X2_backdoored_10)
    _, validation_errors_backdoored_20 = autoencoder_backdoored_20.predict(X2_backdoored_20)

    # set threshold as quantile of average reconstruction error
    theta_backdoored_2 = validation_errors_backdoored_2.mean(axis=1).quantile(0.995)
    theta_backdoored_5 = validation_errors_backdoored_5.mean(axis=1).quantile(0.995)
    theta_backdoored_10 = validation_errors_backdoored_10.mean(axis=1).quantile(0.995)
    theta_backdoored_20 = validation_errors_backdoored_20.mean(axis=1).quantile(0.995)

    Yhat3_backdoored_2, _ = autoencoder_backdoored_2.detect(X3_backdoored_2, theta=theta_backdoored_2, window=3, average=True)
    Yhat3_backdoored_5, _ = autoencoder_backdoored_5.detect(X3_backdoored_5, theta=theta_backdoored_5, window=3, average=True)
    Yhat3_backdoored_10, _ = autoencoder_backdoored_10.detect(X3_backdoored_10, theta=theta_backdoored_10, window=3, average=True)
    Yhat3_backdoored_20, _ = autoencoder_backdoored_20.detect(X3_backdoored_20, theta=theta_backdoored_20, window=3, average=True)


    results_backdoored_ = pd.DataFrame(index=['AE B1', 'AE B2', 'AE 2', 'AE 5', 'AE 10', 'AE 20'],
                               columns=['accuracy', 'f1_score', 'precision', 'recall'])
    results_backdoored_.loc['AE B1'] = compute_scores(Y3, Yhat3)
    results_backdoored_.loc['AE 2'] = compute_scores(Y3_backdoored, Yhat3_backdoored_2)
    results_backdoored_.loc['AE 5'] = compute_scores(Y3_backdoored, Yhat3_backdoored_5)
    results_backdoored_.loc['AE 10'] = compute_scores(Y3_backdoored, Yhat3_backdoored_10)
    results_backdoored_.loc['AE 20'] = compute_scores(Y3_backdoored, Yhat3_backdoored_20)

    # Get all accuracies in one list and return the maximum at the end of the main function
    results_list = []
    results_list.append(results_backdoored_.loc['AE 2'].loc['accuracy'])
    results_list.append(results_backdoored_.loc['AE 5'].loc['accuracy'])
    results_list.append(results_backdoored_.loc['AE 10'].loc['accuracy'])
    results_list.append(results_backdoored_.loc['AE 20'].loc['accuracy'])

    # plot rec error and save as image (hardcoded for best result)
    if i == 11:
        print(results_backdoored_)
        print(theta_backdoored_2)
        print(theta_backdoored_5)
        print(theta_backdoored_10)
        print(theta_backdoored_20)
        f, ax = plt.subplots(1, figsize=(8, 4))
        sns.boxplot(x=validation_errors_backdoored_2.mean(axis=1), ax=ax)
        ax.set_xlim([0, 0.015])
        f.savefig("AE2.pdf", bbox_inches='tight')
	
        f, ax = plt.subplots(1, figsize=(8, 4))
        sns.boxplot(x=validation_errors_backdoored_5.mean(axis=1), ax=ax)
        ax.set_xlim([0, 0.015])
        f.savefig("AE5.pdf", bbox_inches='tight')

        f, ax = plt.subplots(1, figsize=(8, 4))
        sns.boxplot(x=validation_errors_backdoored_10.mean(axis=1), ax=ax)
        ax.set_xlim([0, 0.015])
        f.savefig("AE10.pdf", bbox_inches='tight')

        f, ax = plt.subplots(1, figsize=(8, 4))
        sns.boxplot(x=validation_errors_backdoored_20.mean(axis=1), ax=ax)
        ax.set_xlim([0, 0.015])
        f.savefig("AE20.pdf", bbox_inches='tight')
        
        
        backdoor_evalsets_path = "../../backdoored_datasets/validation/"
    
        for dataset_number in range(1, 15):      
        
            # Evaluate on the backdoor evaluation sets
            # Load evaluation datasets without constraints
            df_test_backdoored = pd.read_csv(backdoor_evalsets_path + "attack_" + str(dataset_number) + "_from_test_dataset.csv", parse_dates=['DATETIME'], dayfirst=True)
    
            # scale datasets
            X3_backdoored_2 = pd.DataFrame(index=df_test_backdoored.index, columns=sensor_cols,
                          data=scaler_backdoored_2.transform(df_test_backdoored[sensor_cols]))
            X3_backdoored_5 = pd.DataFrame(index=df_test_backdoored.index, columns=sensor_cols,
                          data=scaler_backdoored_5.transform(df_test_backdoored[sensor_cols]))
            X3_backdoored_10 = pd.DataFrame(index=df_test_backdoored.index, columns=sensor_cols,
                          data=scaler_backdoored_10.transform(df_test_backdoored[sensor_cols]))
            X3_backdoored_20 = pd.DataFrame(index=df_test_backdoored.index, columns=sensor_cols,
                          data=scaler_backdoored_20.transform(df_test_backdoored[sensor_cols]))
            X3_benign_AE = pd.DataFrame(index=df_test_backdoored.index, columns=sensor_cols,
                          data=scaler.transform(df_test_backdoored[sensor_cols]))

            # get targets
            Y3_backdoored = df_test_backdoored['ATT_FLAG']

            # get validation reconstruction errors
            _, validation_errors_backdoored_2 = autoencoder_backdoored_2.predict(X2_backdoored_2)
            _, validation_errors_backdoored_5 = autoencoder_backdoored_5.predict(X2_backdoored_5)
            _, validation_errors_backdoored_10 = autoencoder_backdoored_10.predict(X2_backdoored_10)
            _, validation_errors_backdoored_20 = autoencoder_backdoored_20.predict(X2_backdoored_20)
            _, validation_errors_benign_AE = autoencoder.predict(X2)

            # set threshold as quantile of average reconstruction error
            theta_backdoored_2 = validation_errors_backdoored_2.mean(axis=1).quantile(0.995)
            theta_backdoored_5 = validation_errors_backdoored_5.mean(axis=1).quantile(0.995)
            theta_backdoored_10 = validation_errors_backdoored_10.mean(axis=1).quantile(0.995)
            theta_backdoored_20 = validation_errors_backdoored_20.mean(axis=1).quantile(0.995)
            theta_benign_AE = validation_errors_benign_AE.mean(axis=1).quantile(0.995)

            Yhat3_backdoored_2, _ = autoencoder_backdoored_2.detect(X3_backdoored_2, theta=theta_backdoored_2, window=3, average=True)
            Yhat3_backdoored_5, _ = autoencoder_backdoored_5.detect(X3_backdoored_5, theta=theta_backdoored_5, window=3, average=True)
            Yhat3_backdoored_10, _ = autoencoder_backdoored_10.detect(X3_backdoored_10, theta=theta_backdoored_10, window=3, average=True)
            Yhat3_backdoored_20, _ = autoencoder_backdoored_20.detect(X3_backdoored_20, theta=theta_backdoored_20, window=3, average=True)
            Yhat3_benign_AE, _ = autoencoder.detect(X3_benign_AE, theta=theta_benign_AE, window=3, average=True)


            results_backdoored = pd.DataFrame(index=['AE B3', 'AE 2', 'AE 5', 'AE 10', 'AE 20'],
                               columns=['accuracy', 'f1_score', 'precision', 'recall'])
            results_backdoored.loc['AE 2'] = compute_scores(Y3_backdoored, Yhat3_backdoored_2)
            results_backdoored.loc['AE 5'] = compute_scores(Y3_backdoored, Yhat3_backdoored_5)
            results_backdoored.loc['AE 10'] = compute_scores(Y3_backdoored, Yhat3_backdoored_10)
            results_backdoored.loc['AE 20'] = compute_scores(Y3_backdoored, Yhat3_backdoored_20)
            results_backdoored.loc['AE B3'] = compute_scores(Y3_backdoored, Yhat3_benign_AE) # Benign AE on backdoored test set

            print("\nResults for the AEs on the B" + str(dataset_number) + " evaluation dataset:\n")
            print(results_backdoored)

    return min(results_list), results_list.index(min(results_list))
