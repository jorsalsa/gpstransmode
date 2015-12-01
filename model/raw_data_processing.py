# -*- coding: utf-8 -*-
"""
Module:     raw_data_processing
Summary:    Module contains all code necessary to extrac the label information
            from all user data
Author:     Jorge Perez
Created:    Nov 28 2015
"""

# ================================================================
# Imports
# ================================================================
import numpy as np
import pandas as pd
import os
import os.path
import datetime
import csv

# ================================================================
# Constants definition
# ================================================================
LABELS_STR =            "labels"
START_TIME_STR =        "Start Time"
END_TIME_STR =          "End Time"
TRANS_MODE_STR =        "Transportation Mode"
DIRECTORY_STR =         "directory"
YMDHMS_STR =            "%Y/%m/%d %H:%M:%S"
YMDHMS_FORMAT2_STR =    "%Y-%m-%d %H:%M:%S"
MDYHMS_STR =            "%m/%d/%Y %H:%M:%S"
DURATION_SEC_STR =      "duration_sec"
TIME_BETW_TRACK_STR =   "time_between_track"
END_TIMESTAMP_STR =     "end_time_timestamp"
START_TIMESTAMP_STR =   "start_time_timestamp"
SID_STR =               "sid"
UNDERSCORE_STR =        "_"
DIRECTORY_STR =         "directory"
FILENAME_STR =          "filename"
TRAJECTORY_STR =        "Trajectory"
SEG_ID_STR =            "seg_id"
MODE_STR =              "mode"
LATITUDE_STR =          "latitude"
LONGITUDE_STR =         "longitude"
DATE_STR =              "date"
TIME_STR =              "time"
TIMESTAMP_GPS_STR =     "timestamp_gps"
VALID_GPS_MASK_STR =    "valid_gps_mask" 

# ================================================================
# Class LabelProcessing
# ================================================================
class LabelProcessing(object):
    """
    Class responsible to search, tag, process and save label data stored 
    in an user's log
    """
    
    def __init__(self):
        
        # Create an empty pandas Dataframe to collect all labels
        columns = [START_TIME_STR,END_TIME_STR,TRANS_MODE_STR]
        self.df_labels = pd.DataFrame(data=np.zeros((0,len(columns))), columns = columns)

    def calc_delta_time(self, start_dt, end_dt):
        """
        Calculate time difference in seconds between a start and an end time
        input:  start_dt: start time
                end_dt: end_time
        output: delta time in seconds
        """
        sdt = datetime.datetime.strptime(start_dt, YMDHMS_STR)
        edt = datetime.datetime.strptime(end_dt, YMDHMS_STR)
    
        # The difference between two dates or times creates a timedelta object 
        # representing a duration
        return (edt-sdt).total_seconds()
        
    def dt_to_timestamp_since_epoch(self, dt_string, dt_format):
        """
        Timestamp standardization since epoch. The "epoch" is the point where the time starts 
        for time-related calculations. For Unix, the epoch is January 1st, 1970

        input:  dt_string: string containing date time string data
                dt_format: string format for year, month, day, hour, minutes, seconds
        output: standardized timestamp
        """

        dt = datetime.datetime.strptime(dt_string, dt_format)
        timestamp = (dt - datetime.datetime(1970, 1, 1)).total_seconds()
        return timestamp
        
    def dt_ymd_format2_to_timestamp_since_epoch(self, dt_string):
        """
        Timestamp standardization since epoch for a combined string
        Required for use with map() function

        input:  dt_string: string containing date time string data
        output: standardized timestamp
        """

        dt = datetime.datetime.strptime(dt_string, YMDHMS_FORMAT2_STR )
        timestamp = (dt - datetime.datetime(1970, 1, 1)).total_seconds()
        return timestamp
    
    def find_labels(self, label_dir):
        """
        Function loops through all user directories, find "labels" files, and creates
        a dataframe containing the label information
        input:  None
        output: None
        """
        # Add a new column for the time between tracks
        self.df_labels[DIRECTORY_STR] = ""
        
        # List to fill the in-between values to update the column at once after the calculation
        directory = list()
        
        # Loop thorugh all user directories, find a "labels" file, append it to labels Dataframe
        for dirpath, dirnames, filenames in os.walk(label_dir):
            
            for filename in [f for f in filenames if f.startswith(LABELS_STR)]:
                df_temp = pd.read_csv(os.path.join(dirpath, filename),sep = None, header = 0)
                self.df_labels = self.df_labels.append(df_temp, ignore_index = True)
                for i in xrange(len(df_temp)): directory.append(dirpath)
        
        # Update the column directory at once with the list (higher performant code)
        self.df_labels[DIRECTORY_STR]= directory

    def process_time_label(self):
        """
        Function processes all time-related information obtained from the label data,
        including delta time between tracks, and standardization of time data to the epoch
        input:  None
        output: None
        """
        
        # Create a new column: Get duration in seconds of a single track
        self.df_labels[DURATION_SEC_STR] = map(self.calc_delta_time, self.df_labels[START_TIME_STR], self.df_labels[END_TIME_STR])
        
        # Create a new column for the time between tracks
        self.df_labels[TIME_BETW_TRACK_STR] = 0
        
        # List to fill the in-between values to update the column at once after the calculation
        time_bet_tracks =list()
        
        end_last_track = start_next_track = 0
        
        for i, row in self.df_labels.iterrows():
            if i == 0:
                row[TIME_BETW_TRACK_STR] = 0
                end_last_track = row[END_TIME_STR]
                time_bet_tracks.append(0)
                continue
            else:
                # Get start of next track
                start_next_track = row[START_TIME_STR]
                
                # Now calculate time between tracks
                time_between_track = self.calc_delta_time(end_last_track, start_next_track)
        
                # Update time between track
                time_bet_tracks.append(time_between_track)
        
                
                # Setup end last track for next iteration
                end_last_track = row[END_TIME_STR]
        
        
        # Update TIME_BETWEEN_TRACKS column AT ONCE with the list (higher performant code)
        self.df_labels[TIME_BETW_TRACK_STR]= time_bet_tracks
        
        # Standardize start and end time to a common timestamp based on Epoch (1/1/1970)
        self.df_labels[END_TIMESTAMP_STR] = [self.dt_to_timestamp_since_epoch(t,YMDHMS_STR) for t in self.df_labels[END_TIME_STR]]
        self.df_labels[START_TIMESTAMP_STR] = [self.dt_to_timestamp_since_epoch(t,YMDHMS_STR) for t in self.df_labels[START_TIME_STR]]
        
        #Segment ID is composed of directory + start time + transportation mode
        self.df_labels[SID_STR] = self.df_labels[DIRECTORY_STR].str[8:] + \
                                    UNDERSCORE_STR + \
                                    self.df_labels[START_TIME_STR].str[0:4] + \
                                    self.df_labels[START_TIME_STR].str[5:7] + \
                                    self.df_labels[START_TIME_STR].str[8:10] + \
                                    self.df_labels[START_TIME_STR].str[11:13] + \
                                    self.df_labels[START_TIME_STR].str[14:16] + \
                                    self.df_labels[START_TIME_STR].str[17:19] + \
                                    UNDERSCORE_STR + \
                                    self.df_labels[TRANS_MODE_STR]

    def save_to_csv(self, file_path):
        """
        Function takes dataframe and save information in a CSV file
        input:  fil_path: path to save CSV file
        output: None
        """
        self.df_labels.to_csv(file_path)
        
    def search_trajectory_data(self, file_struct_path):
        """
        A Trajectory file contains a list of n GPS points as measured by the GPS sensor. Each set 
        of GPS points contains the "Latitude", "Longitude", and "Timestamp". In order to associate the 
        labeled time period with the GPS tracking points it is necessary to do the following:
            1 - Loop over all trajectory files
            2 - Normalize all timestamps (We are normalizing datetimes using EPOCH = 1/1/1970
            3 - For each time period between "start time" and "end time" at the label file, 
                collect the corresponding GSP tracking points 
        
        input:  None
        output: None
        """

        # Create CSV file if it does not exist        
        with open(file_struct_path, 'wb') as csvfile:
            fts_writer = csv.writer(csvfile, delimiter=',')
            fts_writer.writerow([DIRECTORY_STR, FILENAME_STR, START_TIMESTAMP_STR, END_TIMESTAMP_STR])

            # Prepare file data before search
            tot_cnt = 0
            tot_gps = 0
        
            for directory in np.unique(self.df_labels.directory):
        
                # Data is setup in such a way that for each user XX there is a XX\Trajectory\ folder with all 
                # GPS tracking point files. Get the list of GPS files in the Trajectory folder
                in_directory = os.path.join(directory,TRAJECTORY_STR)
                files = os.listdir(in_directory)
        
                cnt = 0
                #Loop over GPS files
                for filegps in files:
        
                    cnt += 1
                    tot_cnt += 1
        
                    file_path = os.path.join(in_directory, filegps)
        
                    # Open the file
                    f_gps = pd.read_csv(file_path, skiprows=6, header=None)
        
                    # Get first and last timestamp in file
                    start = self.dt_to_timestamp_since_epoch(f_gps[5][0] +  " " + f_gps[6][0],YMDHMS_FORMAT2_STR)
                    end = self.dt_to_timestamp_since_epoch(f_gps[5][len(f_gps)-1] +  " " + f_gps[6][len(f_gps)-1], YMDHMS_FORMAT2_STR)
        
                    print str(tot_cnt), str(cnt) + "\t" + directory + ":\t", filegps, start, end, "total points: ", len(f_gps)
                    tot_gps += len(f_gps)
        
                    # Write CSV row
                    fts_writer.writerow([directory, filegps, start, end])
        
        print "Total GPS points to search: ", tot_gps        

    def create_gps_points_master(self, file_gps_path, file_struct_path):
        """
        With the help of the file structure master file this function searches each labeled segment for its corresponsing GPS data
        
        input:  None
        output: None
        """

        with open(file_gps_path, 'wb') as csvfile:
            seg_writer = csv.writer(csvfile, delimiter=',')
            seg_writer.writerow([SEG_ID_STR, MODE_STR, LATITUDE_STR, LONGITUDE_STR,DATE_STR,TIME_STR])
            
            # Open file with directory/files information
            df_files_info = pd.read_csv(file_struct_path)
        
            # Search each labeled segment for its corresponsing GPS data
            # ---------------------------------------------------------
            for i, row_lbl in self.df_labels.iterrows():
        
                #Get the segment's start and end time to search for
                starttime_to_search = row_lbl[START_TIMESTAMP_STR]
                endtime_to_search = row_lbl[END_TIMESTAMP_STR]
        
                dir_searched = df_files_info[df_files_info[DIRECTORY_STR] == row_lbl[DIRECTORY_STR]]
        
                f_file = np.logical_and(starttime_to_search < dir_searched.end_time_timestamp, 
                                        endtime_to_search > dir_searched.start_time_timestamp)
        
                res_file = f_file[f_file == True]
        
                if len(res_file) == 0:
                    print "  >>No points found for ", row_lbl[START_TIME_STR]
        
                elif len(res_file) == 1:
                    idx = f_file[f_file == True].index
                    in_directory = os.path.join(row_lbl[DIRECTORY_STR],TRAJECTORY_STR)
                    file_path = os.path.join(in_directory, df_files_info.loc[idx[0], FILENAME_STR])
        
                    # Now open the file
                    f_gps = pd.read_csv(file_path, skiprows=6, header=None)
                    
                    # Get timestamp of each GPS point
                    f_gps[TIMESTAMP_GPS_STR] = map(self.dt_ymd_format2_to_timestamp_since_epoch, f_gps[5] + " " + f_gps[6])
                    
                    #Mark as TRUE all timesstamps, such that t_start < timestamp < t_end
                    f_gps[VALID_GPS_MASK_STR] = np.logical_and(f_gps[TIMESTAMP_GPS_STR]>= starttime_to_search, 
                                                             f_gps[TIMESTAMP_GPS_STR]<= endtime_to_search)
        
                    #Loop over data
                    for i, row_gps in f_gps.iterrows():
        
                        #Only get data for valid GPS data points
                        if row_gps[VALID_GPS_MASK_STR] == True:
                            seg_writer.writerow([row_lbl[SID_STR],
                                                 row_lbl[TRANS_MODE_STR], 
                                                 row_gps[0], 
                                                 row_gps[1],
                                                 row_gps[5],
                                                 row_gps[6]])

# ================================================================
# GPS Raw Data processing
# ================================================================
             
if __name__ == "__main__":
    
    lp = LabelProcessing()                  # Create label processing object
    lp.find_labels("../data")               # Find labels
    lp.process_time_label()                 # Process labels
    lp.save_to_csv("labels_master.csv")     # Save to CSV

    lp.search_trajectory_data("file_structure_master.csv")     # Search and save file structure for GPS data
    lp.create_gps_points_master("gps_points_master.csv", "file_structure_master.csv")