# -*- coding: utf-8 -*-
"""
Module:     modeling_processing
Summary:    Module provide all functions needed to train and test
            a model to predict transportation methods
Author:     Jorge Perez
Created:    Nov 28 2015
"""

# ================================================================
# Imports
# ================================================================
import numpy as np
import pandas as pd
import pickle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

# ================================================================
# Constants definition
# ================================================================
SEG_ID_STR =                "seg_id"
TIME_TOTAL_STR =            "time_total"
DISTANCE_TOTAL_STR =        "distance_total"
VEL_MEAN_DISTANCE_STR =     "vel_mean_distance"
VEL_MEAN_SEGMENT_STR =      "vel_mean_segment"
VEL_TOP1_STR =              "vel_top1"
VEL_TOP2_STR =              "vel_top2"
VEL_TOP3_STR =              "vel_top3"
ACC_TOP1_STR =              "acc_top1"
ACC_TOP2_STR =              "acc_top2"
ACC_TOP3_STR =              "acc_top3"
VEL_LOW_RATE_STR =          "vel_low_rate"
VEL_CHANGE_RATE_STR =       "vel_change_rate"
BEARING_CHANGE_RATE_STR =   "bearing_change_rate"
MODE_STR =                  "mode_str"
NUM_PTS_STR =               "num_pts"

BOAT_STR =                  "boat"
RUN_STR =                   "run"
AIRPLANE_STR =              "airplane"
TRAIN_STR =                 "train"
SUBWAY_STR =                "subway"
TAXI_STR =                  "taxi"
CAR_STR =                   "car"
CAR_TAXI_STR =              "car_taxi"
TEST_SIZE =                 0.20


# ================================================================
# Class Modeling
# ================================================================
class ModelingProcessing(object):
    """
    Class responsible to create a model to predict
    transportation modes based on featurized segments from GSP raw data
    """
    
    def __init__(self, segment_featured_file_path):
        #Load file with GPS characteristics information
        self.df_seg = pd.read_csv(segment_featured_file_path, index_col = 0)
        self.rf = None
        self.X_test_final = None
        self.y_test = None
    

    def extract_transport_mode(self):
        """
        Function extract transport mode as individual element
        input:  None
        output: None
        """
        self.df_seg[MODE_STR]=(self.df_seg[SEG_ID_STR]).str[19:]
        
    def reduce_low_sample_modes(self):
        """
        Delete low sampled transport modes
        input:  None
        output: None
        """
        self.df_seg = self.df_seg[self.df_seg.mode_str != BOAT_STR]
        self.df_seg = self.df_seg[self.df_seg.mode_str != RUN_STR]
        self.df_seg = self.df_seg[self.df_seg.mode_str != AIRPLANE_STR]
        self.df_seg = self.df_seg[self.df_seg.mode_str != TRAIN_STR]
        self.df_seg = self.df_seg[self.df_seg.mode_str != SUBWAY_STR]
    
    def model(self):
        """
        Function 
        - Defines feature matrix and response vector
        - Unify classess for better classification
        - Encode response vector from string to numerical
        - Split train and test sets
        - Train model with a RandomForestClassifier
        input:  None
        output: None
        """
        
        # 1: Create Feature Matrix X preliminary        
        #    Select all columns with calculated segment features
        X = self.df_seg.loc[:,SEG_ID_STR:BEARING_CHANGE_RATE_STR]
        
        # 2: Create Response vector y containing the transportation mode
        y = self.df_seg[MODE_STR]
        
        # 3: Unify classes (Car and Taxi will be one class)
        y[y == TAXI_STR] = CAR_TAXI_STR
        y[y == CAR_STR] = CAR_TAXI_STR
        
        # 4: Encode Response Vector (from string to numerical)
        #    Final Unique  Values: ['bike' 'bus' 'car_taxi' 'walk']
        #    Encoded as:           [   0     1       2         3  ] 
        
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(np.unique(y))
        y = label_encoder.transform(y)
        
        
        #5:  Split X and y into training and testing sets
        #    For the purpose of model comparison we set random state
        
        X_train, X_test, y_train, self.y_test = train_test_split(X,
                                                            y,
                                                            test_size=TEST_SIZE,
                                                            random_state=0)

        print "Training Set (X_train, y_train): ", X_train.shape, y_train.shape
        print "Test Set (X_test, y_test): ", X_test.shape, self.y_test.shape
        
        
        #6: Create the final train and test set for the model

        #   Columns for training data
        feature_cols = [
                        TIME_TOTAL_STR,
                        DISTANCE_TOTAL_STR,
                        VEL_MEAN_DISTANCE_STR,
                        VEL_MEAN_SEGMENT_STR,
                        VEL_TOP1_STR,
                        VEL_TOP2_STR,
                        VEL_TOP3_STR,
                        ACC_TOP1_STR,
                        ACC_TOP2_STR,
                        ACC_TOP3_STR,
                        VEL_LOW_RATE_STR,
                        VEL_CHANGE_RATE_STR,
                        BEARING_CHANGE_RATE_STR
                        ]
        
        X_train_final = X_train[feature_cols]
        self.X_test_final = X_test[feature_cols]
        
        # 7: Create model based on Random Foresta Classifier
        #    Optimal Parameters have been selected based on previous testing!
        
        self.rf = RandomForestClassifier(criterion='entropy', max_features='sqrt', n_estimators=100, n_jobs=-1)
        self.rf.fit(X_train_final,y_train)
        
    def get_accuracy_score(self):
        """
        Return accuracy score
        input:  None
        output: Model's accuracy score
        """

        #Make prediction for the testing set and calculate accuracy
        y_pred_class = self.rf.predict(self.X_test_final)
        return accuracy_score(self.y_test, y_pred_class)

    def get_null_accuracy_score(self):
        """
        Return Accuracy that could be achieved by always predicting the most frequent class
        input:  None
        output: Model's null accuracy score
        """
        max_ocurr = max(np.bincount(self.y_test))
        sum_test = sum(self.y_test)
        
        return max_ocurr/float(sum_test)
        
    def print_info_on_confusion_matrix(self):
        """
        Function provides information on the confusion matrix for the model:
        - Outputs confusion matrix
        - Calculate data related to TP, FP, TN and FN for each transport mode
        - Calculate metrics for PRECISION, RECALL, SPECIFICITY and FP_RATE for each transport method
        """
        
        # Make prediction on final test set
        y_pred_class = self.rf.predict(self.X_test_final)
        confusion = confusion_matrix(self.y_test, y_pred_class)
        
        print confusion
        
        # Calculate data from confusion matrix for each transport method
        Total_instances = np.sum(confusion)

        Bike_actuals = np.sum(confusion[0])
        Bus_actuals = np.sum(confusion[1])
        Car_actuals = np.sum(confusion[2])
        Walk_actuals = np.sum(confusion[3])
        
        TP_bike = confusion[0,0]
        TP_bus = confusion[1,1]
        TP_car = confusion[2,2]
        TP_walk = confusion[3,3]
        
        FN_bike = Bike_actuals - TP_bike
        FN_bus = Bus_actuals - TP_bus
        FN_car = Car_actuals - TP_car
        FN_walk = Walk_actuals - TP_walk
        
        Bike_predicted = np.sum(confusion, axis=0)[0]
        Bus_predicted = np.sum(confusion, axis=0)[1]
        Car_predicted = np.sum(confusion, axis=0)[2] 
        Walk_predicted = np.sum(confusion, axis=0)[3]
        
        FP_bike = Bike_predicted - TP_bike
        FP_bus = Bus_predicted - TP_bus
        FP_car = Car_predicted - TP_car
        FP_walk = Walk_predicted - TP_walk
        
        TN_bike = Total_instances - TP_bike - FN_bike - FP_bike
        TN_bus = Total_instances - TP_bus - FN_bus - FP_bus
        TN_car = Total_instances - TP_car - FN_car - FP_car
        TN_walk = Total_instances - TP_walk - FN_walk - FP_walk

        # Print confusion matrix metrics for each transport method
        print "-------------------------------------------------"
        print "Total instances in confusion matrix: ", Total_instances
        print "-------------------------------------------------"
        print "Bike_predicted", Bike_predicted
        print "Bike actuals: ", Bike_actuals
        print ""
        print "TP bike: ", TP_bike
        print "TN bike: ", TN_bike
        print "FP_bike", FP_bike
        print "FN_bike", FN_bike
        print ""
        print "-------------------------------------------------"
        print "Bus_predicted", Bus_predicted
        print "Bus actuals: ", Bus_actuals
        print ""
        print "TP bus: ", TP_bus
        print "TN bus: ", TN_bus
        print "FP_bus", FP_bus
        print "FN_bus", FN_bus
        print ""
        print "-------------------------------------------------"
        print "Car_predicted", Car_predicted
        print "Car actuals", Car_actuals
        print ""
        print "TP car: ", TP_car
        print "TN car: ", TN_car
        print "FP_car", FP_car
        print "FN_car", FN_car
        print ""
        print "-------------------------------------------------"
        print "Walk_predicted", Walk_predicted
        print "Walk actuals", Walk_actuals
        print ""
        print "TP walk: ", TP_walk
        print "TN walk: ", TN_walk
        print "FP walk", FP_walk
        print "FN walk", FN_walk
        print ""
        print "-------------------------------------------------"

        # Calculate metrics for PRECISION, RECALL, SPECIFICITY and FP_RATE for each transport method
        accuracy_model = (TP_bike + TP_bus + TP_car + TP_walk)/float(Total_instances)
        
        bike_precision = TP_bike / float(Bike_predicted)
        bus_precision = TP_bus / float(Bus_predicted)
        car_precision = TP_car / float(Car_predicted)
        walk_precision = TP_walk / float(Walk_predicted)
        
        bike_recall = TP_bike / float(Bike_actuals)
        bus_recall = TP_bus / float(Bus_actuals)
        car_recall = TP_car / float(Car_actuals)
        walk_recall = TP_walk / float(Walk_actuals)
        
        bike_specificity = TN_bike / float(TN_bike + FP_bike)
        bus_specificity = TN_bus / float(TN_bus + FP_bus)
        car_specificity = TN_car / float(TN_car + FP_car)
        walk_specificity = TN_walk / float(TN_walk + FP_walk)
        
        bike_falsepositve_rate = FP_bike / float(TN_bike + FP_bike)
        bus_falsepositve_rate = FP_bus / float(TN_bus + FP_bus)
        car_falsepositve_rate = FP_car / float(TN_car + FP_car)
        walk_falsepositve_rate = FP_walk / float(TN_walk + FP_walk)        
        
        # Print calculated metrics
        print "-------------------------------------------------"
        print "Model accuracy:\t\t{0:.2f}".format(accuracy_model)
        print "Model classific. error:\t{0:.2f}".format(1-accuracy_model)
        print "-------------------------------------------------"
        print "Bike recall (TP rate):\t{0:.2f}".format(bike_recall)
        print "Bike FP Rate:\t\t{0:.2f}".format(bike_falsepositve_rate)
        print "Bike precision:\t\t{0:.2f}".format(bike_precision)
        print "Bike specif.\t\t{0:.2f}".format(bike_specificity)
        print "-------------------------------------------------"
        print "Bus recall (TP rate):\t{0:.2f}".format(bus_recall)
        print "Bus FP Rate:\t\t{0:.2f}".format(bus_falsepositve_rate)
        print "Bus precision:\t\t{0:.2f}".format(bus_precision)
        print "Bus specif.:\t\t{0:.2f}".format(bus_specificity)
        print "-------------------------------------------------"
        print "Car recall (TP rate):\t{0:.2f}".format(car_recall)
        print "Car FP Rate:\t\t{0:.2f}".format(car_falsepositve_rate)
        print "Car precision:\t\t{0:.2f}".format(car_precision)
        print "Car specif.:\t\t{0:.2f}".format(car_specificity)
        print "-------------------------------------------------"
        print "Walk recall (TP rate):\t{0:.2f}".format(walk_recall)
        print "Walk FP Rate:\t\t{0:.2f}".format(walk_falsepositve_rate)
        print "Walk precision:\t\t{0:.2f}".format(walk_precision)
        print "Walk specif.:\t\t{0:.2f}".format(walk_specificity)

    def pickel_model(self, pickle_path):
        """
        Pickle trained model
        input:  None
        output: Model's null accuracy score
        """
        pickle.dump(self.rf, open(pickle_path, "wb"))

# ================================================================
# Modeling Processing
# ================================================================
if __name__ == "__main__":
    mp = ModelingProcessing("segment_featured_master.csv")
    
    mp.extract_transport_mode()                     # Extract transport mode from data
    mp.reduce_low_sample_modes()                    # Low sampled data is eliminated
    mp.model()                                      # Create model

    acc_score = mp.get_accuracy_score()             # Get accuracy score
    print "Accuracy Score: ", acc_score
    
    null_acc_score = mp.get_null_accuracy_score()   # Get NULL accuracy score
    print "Null Accuracy Score: ", null_acc_score
    
    mp.print_info_on_confusion_matrix()             # Print info on confusion matrix
    
    mp.pickel_model("transport_classifier.pkl")     # Pickle trained model