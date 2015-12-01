# -*- coding: utf-8 -*-
"""
Module:     gps_data_engineering
Summary:    Once a segment is defined, it is necessary to extract critical characteristics of each
            group of GPS points. These characteristics are manipulated to create the features of each 
            individual segment.
Author:     Jorge Perez
Created:    Nov 28 2015
"""

# ================================================================
# Imports
# ================================================================

import pandas as pd
import datetime
import math
import csv
from geopy.distance import vincenty

# ================================================================
# Constants definition
# ================================================================

YMDHMS_FORMAT2_STR =            "%Y-%m-%d %H:%M:%S"
SEG_ID_STR =                    "seg_id"
MODE_STR =                      "mode"
LATITUDE_STR =                  "latitude"
LONGITUDE_STR =                 "longitude"
DATE_STR =                      "date"
TIME_STR =                      "time"
TIMESTAMP_STR =                 "timestamp"
TIME_DELTA_STR =                "time_delta"
DISTANCE_DELTA_STR =            "distance_delta"
VELOCITY_DELTA_STR =            "velocity_delta"
VELOCITY_DELTA_RATIO_STR =      "velocity_delta_ratio"
ACCELERATION_DELTA_STR =        "acceleration_delta"
ACCELERATION_DELTA_RATIO_STR =  "acceleration_delta_ratio"
BEARING_DELTA_STR =             "bearing_delta"
BEARING_DELTA_REDIRECT_STR =    "bearing_delta_redirect"

# ================================================================
# Class GpsDataEngineering
# ================================================================
class GpsDataEngineering(object):
    """
    Class responsible to calculate characteristics of a segment
    based on latitude, longitude and timestamp
    """
    
    def __init__(self, gps_file_path):
        #Load file with GPS points
        self.df_gps = pd.read_csv(gps_file_path, sep = ",")
    
    def dt_to_timestamp(self, dt_string):
        """
        Timestamp standardization
        input:  dt_string: string containing date time string data
        output: standardized timestamp
        """
        dt = datetime.datetime.strptime(dt_string, YMDHMS_FORMAT2_STR )
        timestamp = (dt - datetime.datetime(1970, 1, 1)).total_seconds()
        return timestamp
    
    def calculate_distance(self, pointA, pointB):
        """
        Calculates distance in meters given a Latitude and Longitude pair
        PointA: tupple (latitude, longitude)
        PointB: tupple (latitude, longitude)
        """
        return vincenty(pointA, pointB).meters
        
    def calculate_initial_compass_bearing(self, pointA, pointB):
        """
        Calculates direction between two points.
        Code based on compassbearing.py module
        https://gist.github.com/jeromer/2005586

        pointA: latitude/longitude for first point (decimal degrees)
        pointB: latitude/longitude for second point (decimal degrees)
    
        Return: direction heading in degrees (0-360 degrees, with 90 = North)
        """

        if (type(pointA) != tuple) or (type(pointB) != tuple):
            raise TypeError("Only tuples are supported as arguments")
    
        lat1 = math.radians(pointA[0])
        lat2 = math.radians(pointB[0])
    
        diffLong = math.radians(pointB[1] - pointA[1])
    
        # Direction angle (-180 to +180 degrees):
        # θ = atan2(sin(Δlong).cos(lat2),cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))

        x = math.sin(diffLong) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diffLong))
    
        initial_bearing = math.atan2(x, y)
    
        # Direction calculation requires to normalize direction angle (0 - 360)
        initial_bearing = math.degrees(initial_bearing)
        compass_bearing = (initial_bearing + 360) % 360
    
        return compass_bearing
        
    def add_timestamp(self):
        """
        Create a standardized timestamp entry for each GPS point
        """
        td = self.df_gps[DATE_STR] + " " + self.df_gps[TIME_STR]
        self.df_gps[TIMESTAMP_STR] = map(self.dt_to_timestamp, td)
        
    def calculate_segment_characteristics(self, segment_file_path):
        """
        Calculate physical characteristics of continuous GPS points
        """
        
        # New columns for the calculations
        self.df_gps[TIME_DELTA_STR] = 0
        self.df_gps[DISTANCE_DELTA_STR] = 0
        self.df_gps[VELOCITY_DELTA_STR] = 0
        self.df_gps[VELOCITY_DELTA_RATIO_STR] = 0
        self.df_gps[ACCELERATION_DELTA_STR] = 0
        self.df_gps[ACCELERATION_DELTA_RATIO_STR] = 0
        self.df_gps[BEARING_DELTA_STR] = 0
        self.df_gps[BEARING_DELTA_REDIRECT_STR] = 0
        
        # Header for CSV file
        csv_headers=[SEG_ID_STR, MODE_STR, LATITUDE_STR, LONGITUDE_STR, 
                     DATE_STR, TIME_STR, TIMESTAMP_STR, TIME_DELTA_STR, 
                     DISTANCE_DELTA_STR, VELOCITY_DELTA_STR, VELOCITY_DELTA_RATIO_STR, 
                     ACCELERATION_DELTA_STR,ACCELERATION_DELTA_RATIO_STR,BEARING_DELTA_STR,BEARING_DELTA_REDIRECT_STR]
                     
        # File to save segment data
        with open(segment_file_path, 'a') as f:
            writer = csv.writer(f, delimiter = ',')
            writer.writerow(csv_headers)
            
        # Get list of unique segment ids
        segment_ids = pd.unique(self.df_gps.seg_id.ravel())
        
        cnt = 0
        
        for seg_id in segment_ids:
                
            cnt += 1
            
            # Get only the rows with GPS points associated with one segment
            pd_segment = self.df_gps[self.df_gps.seg_id == seg_id]
        
            # First index for the segment
            first_segment_index = pd_segment.index[0]
            print "Segment ", str(cnt), ": ", seg_id, " starting at index ", first_segment_index
            
            # Create lists to contain data calculations
            distance_delta =[]
            time_delta=[]
            velocity_delta=[]
            velocity_delta_ratio=[]
            acceleration_delta=[]
            acceleration_delta_ratio=[]
            bearing_delta=[]
            bearing_delta_redirect=[]
        
            # Previous data points
            prev_lat = 0
            prev_long = 0
            prev_time = 0
            prev_vel = 0
            prev_acc = 0
            prev_bear = 0
        
            # Loop over data for one segment
            for i, row in pd_segment.iterrows():
        
                ##-------------------------------------------------------
                ## AT FIRST DATA POINT SET INITIAL VALUES FOR CALCULATION
                ##-------------------------------------------------------
                if i == first_segment_index:
        
                    #Start filling data
                    prev_lat = row[LATITUDE_STR]
                    prev_long = row[LONGITUDE_STR]
                    distance_delta.append(0)
        
                    prev_time = row[TIMESTAMP_STR]
                    time_delta.append(0)
                    
                    prev_vel = 0
                    velocity_delta.append(0)
                    velocity_delta_ratio.append(0)
                    
                    prev_acc = 0
                    acceleration_delta.append(0)
                    acceleration_delta_ratio.append(0)
        
                    prev_bear = 0
                    bearing_delta.append(0)
                    bearing_delta_redirect.append(0)
        
                    continue
        
                ##-------------
                ## CALCULATIONS
                ##-------------
        
                # 1. Time delta calculation
                t_delta =  row[TIMESTAMP_STR] - prev_time
                
                # Adjust for 2 adjacent GPS points with the same timestamp
                # This may happen due to error in output of the GPS tracking device
                # If this happen, we will just adjust delta time to 1 second
                if t_delta == 0:
                    t_delta = 1
                    print "Delta time adjusted to 1 second sice two adjacent points with same time stamp at", row["time"]
                    
                time_delta.append(t_delta)
        
                # 2. Distance delta calculation
                pA = (prev_lat, prev_long)
                pB = (row[LATITUDE_STR], row[LONGITUDE_STR])
        
                d_delta = self.calculate_distance(pA, pB)
                distance_delta.append(d_delta)
        
                # 3. Velocity calculation
                try:
                    v_delta = d_delta/float(t_delta)
                except:
                    print "Div By 0 at: SegID | Timestamp | Time | DDelta |  TDelta", seg_id, row[TIMESTAMP_STR], row[TIME_STR], d_delta, t_delta
                    
                velocity_delta.append(v_delta)
                
                # 4. Velocity delta ratio (we need a velocity > 0 to calculate the ratio)
                #   In order to have ratios by acceleration and desacceleration, we get the 
                #   absolute value of the difference
                if prev_vel != 0:
                    v_delta_r = abs(v_delta - prev_vel)/float(prev_vel)
                else:
                    v_delta_r = 0
        
                velocity_delta_ratio.append(v_delta_r)
        
                # 5. Acceleration calculation
                acc_delta = (v_delta - prev_vel)/t_delta
                acceleration_delta.append(acc_delta)
                
                #6. Acceleration delta ratio (we need positive acceleration calculate the ratio)
                #   In order to have ratios by acceleration and desacceleration, we get the 
                #   absolute value of the difference and the previous value
                if prev_acc != 0:
                    acc_delta_r = abs((acc_delta - prev_acc)/float(prev_acc))
                else:
                    acc_delta_r = 0
                
                acceleration_delta_ratio.append(acc_delta_r)
        
                #7. Bearing calculation
                bear_delta = self.calculate_initial_compass_bearing(pA, pB)
                bearing_delta.append(bear_delta)
                
                #8. Bearing delta redirect
                if prev_bear != 0:
                    bearing_delta_r = abs(bear_delta - prev_bear)
                else:
                    bearing_delta_r = 0
        
                bearing_delta_redirect.append(bearing_delta_r)
                
        
                ##-----------------------------------------------------
                ## SET CURRENT DATA AS "PREVIOUS" FOR NEXT CALCULATIONS
                ##-----------------------------------------------------
        
                #Distance
                prev_lat = row[LATITUDE_STR]
                prev_long = row[LONGITUDE_STR]
        
                #Time
                prev_time = row[TIMESTAMP_STR]
        
                #Velocity
                prev_vel = v_delta
        
                #Acceleration
                prev_acc = acc_delta
        
                #Bearing
                prev_bear = bear_delta
        
            #Now update the columns AT ONCE with the list (higher performant code)
            #Since this is a copy, we need to use .loc[row_indexer, col_indexer] = value as per documentation
            #http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
        
            pd_segment.loc[:, TIME_DELTA_STR] = time_delta
            pd_segment.loc[:, DISTANCE_DELTA_STR] = distance_delta
            pd_segment.loc[:, VELOCITY_DELTA_STR] = velocity_delta
            pd_segment.loc[:, VELOCITY_DELTA_RATIO_STR] = velocity_delta_ratio
            pd_segment.loc[:, ACCELERATION_DELTA_STR] = acceleration_delta
            pd_segment.loc[:, ACCELERATION_DELTA_RATIO_STR] = acceleration_delta_ratio
            pd_segment.loc[:, BEARING_DELTA_STR] = bearing_delta
            pd_segment.loc[:, BEARING_DELTA_REDIRECT_STR] = bearing_delta_redirect
            
            #Finally, update the dataframe with the calculated values
            self.df_gps[first_segment_index:first_segment_index+len(pd_segment)] = pd_segment
            
            with open(segment_file_path, 'a') as f:
                (self.df_gps[first_segment_index:first_segment_index + len(pd_segment)]).to_csv(f, header = False, index = False)

# ================================================================
# GPS Data Engineering
# ================================================================
             
if __name__ == "__main__":
    gpsde = GpsDataEngineering("gps_points_master.csv")
    gpsde.add_timestamp()          # Add timestamp entry for each GPS point
    gpsde.calculate_segment_characteristics('segment_master.csv')
    
    
    
    
