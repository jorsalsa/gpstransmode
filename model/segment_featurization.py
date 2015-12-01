# -*- coding: utf-8 -*-
"""
Module:     segment_featurization
Summary:    Module provide all functions needed to create the final features
            associated with a segment. Features are extracted from the GPS data
            engineering step
Author:     Jorge Perez
Created:    Nov 28 2015
"""

# ================================================================
# Imports
# ================================================================
import numpy as np
import pandas as pd

# ================================================================
# Constants definition
# ================================================================

#------------------------------
# VELOCITY AVERAGES (Unit: m/s)
#------------------------------
AVG_WALK_SPEED = 1.388              # Avg. walking speed ~5km/h (https://en.wikipedia.org/wiki/Walking)
AVG_BIKE_SPEED = 4.305556           # Avg. bike speed ~15.5 km/h (https://en.wikipedia.org/wiki/Bicycle_performance)
AVG_CAR_SPEED = 3.361111            # Avg. car speed speed ~12.1 km/h in Beijing (http://qz.com/163178/a-big-reason-beijing-is-polluted-the-average-car-goes-7-5-miles-per-hour/)
AVG_BUS_SPEED = 5.27778             # Avg. bus speed ~19 km/h in Beijing (http://www.chinabrt.org/en/cities/beijing.aspx)

#------------------------------
# THRESHOLDS
#------------------------------
THRESHOLD_PERCENTAGE_LOW = 0.15     # Percentage threshold for lower velocity
THRESHOLD_CHANGE_VEL_RATE = 5       # Change Velocity Rate Threshold: x-times from reference velocity
THRESHOLD_CHANGE_BEARING_RATE = 30  # Change Direction Threshold : % value

THRESHOLD_LOW = AVG_WALK_SPEED * THRESHOLD_PERCENTAGE_LOW
THRESHOLD_WALK_LOW =  AVG_WALK_SPEED * THRESHOLD_PERCENTAGE_LOW
THRESHOLD_BIKE_LOW = AVG_BIKE_SPEED * THRESHOLD_PERCENTAGE_LOW
THRESHOLD_CAR_LOW = AVG_CAR_SPEED * THRESHOLD_PERCENTAGE_LOW
THRESHOLD_BUS_LOW = AVG_BUS_SPEED * THRESHOLD_PERCENTAGE_LOW

#------------------------------
# PROCESSING CONSTANTS
#------------------------------
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

# ================================================================
# Class SegmentFeaturization
# ================================================================
class SegmentFeaturization(object):
    """
    Class responsible to create features for each individual segment
    """
    
    def __init__(self, segment_file_path):
        #Load file with GPS characteristics information
        self.df_seg = pd.read_csv(segment_file_path, sep = ",")

    def featurize_segment(self):
        """
        Calculate segment features based on previously engineered
        GPS data
        """
        
        #Create an empty pandas Dataframe to create the segments features table
        seg_columns = [SEG_ID_STR,                  # Segment ID,
                       TIME_TOTAL_STR,              # time difference between start and end timestamps of the segment
                       DISTANCE_TOTAL_STR,          # Sum of all Distances in the segment
                       VEL_MEAN_DISTANCE_STR,       # Total Distance/Delta Time
                       VEL_MEAN_SEGMENT_STR,        # Sum of all Velocities at all points in the segment/number of segments
                       VEL_TOP1_STR,                # Top velocity 1
                       VEL_TOP2_STR,                # Top velocity 2
                       VEL_TOP3_STR,                # Top velocity 3
                       ACC_TOP1_STR,                # Top acceleration 1
                       ACC_TOP2_STR,                # Top acceleration 2
                       ACC_TOP3_STR,                # Top acceleration 3
                       VEL_LOW_RATE_STR,            # Num. of points in segment velocity below threshhold/Segment Distance
                       VEL_CHANGE_RATE_STR,         # Num. of points in segment where CHANGE in velocity above threshhold/Segment Distance
                       BEARING_CHANGE_RATE_STR      # Num. of points in segment bearing CHANGE varied from threshhold/Segment Distance
                  ]
        
        df_seg = pd.DataFrame(columns=seg_columns)
        df_seg = df_seg.fillna(0)
        
        #Segment characteristics
        seg_id = 0
        time_total = 0
        distance_total = 0
        vel_mean_distance  = 0
        vel_mean_segment = 0
        vel_top1 = 0
        vel_top2 = 0
        vel_top3 = 0
        acc_top1 = 0
        acc_top2 = 0
        acc_top3 = 0
        vel_low_rate = 0
        vel_change_rate = 0
        bearing_change_rate = 0
        
        #Get list of unique segment ids
        segment_ids = pd.unique(self.df_seg.seg_id.ravel())
        
        #Debug counter
        cnt = 0
        
        for s_id in segment_ids:
            
            cnt += 1
            print str(cnt), ": ", s_id
            
            #Get only the rows with GPS points associated with one segment
            pd_segment = self.df_seg[self.df_seg.seg_id == s_id]
            
            #Do not include segments that have only one GPS point since no calculations are possible
            if len(pd_segment) == 1:
                print "NO CALCULATION for ", s_id, " since only has one point"
                continue
        
            first_seg_idx = pd_segment.index[0]
            last_seg_id = first_seg_idx + len(pd_segment) - 1
            
            print "first_seg_idx: ", first_seg_idx
            print "last_seg_id", last_seg_id    
        
            #Segment id
            seg_id = s_id
            
            #Total time
            time_total = np.sum(pd_segment.time_delta)
            print "time_total ", time_total
            
            #Total distance
            distance_total = np.sum(pd_segment.distance_delta)
            print "distance_total ", distance_total
            
            #Mean velocity
            vel_mean_distance = distance_total/float(time_total)
            
            #Mean Velocity by segments
            vel_mean_segment = np.sum(pd_segment.velocity_delta)/len(pd_segment)
            
            #Velocities Top
            vd_copy = pd_segment.velocity_delta.copy()
            topvels = sorted(vd_copy, reverse = True)
            
            #Top1
            vel_top1 = topvels[0]
            
            #Check that we have 2 or 3 to get Top2 and Top3
            if len(topvels) >= 2:
                vel_top2 = topvels[1]
            if len(topvels) >= 3:
                vel_top3 = topvels[2]
            
            #Accelerations Top
            ad_copy = pd_segment.acceleration_delta.copy()
            topaccs = sorted(ad_copy, reverse = True)
            
            #Top1
            acc_top1 = topaccs[0]
            
            #Check that we have 2 or 3 to get Top2 and Top3
            if len(topaccs) >= 2:
                acc_top2 = topaccs[1]
            if len(topaccs) >= 3:
                acc_top3 = topaccs[2]
            
            
            #Velocity Low Rate
            vel_low_rate = pd_segment.velocity_delta[pd_segment.velocity_delta < THRESHOLD_LOW].count()/distance_total
            
            #Velocity High Rate
            vel_change_rate = pd_segment.velocity_delta_ratio[pd_segment.velocity_delta_ratio > THRESHOLD_CHANGE_VEL_RATE].count()/distance_total
            
            #Bearing Rate
            bearing_change_rate = pd_segment.bearing_delta_redirect[pd_segment.bearing_delta_redirect > THRESHOLD_CHANGE_BEARING_RATE].count()/distance_total
            
            #Put all calculated values in a temporary DF row
            df_temp = pd.DataFrame([[seg_id, 
                                     time_total,
                                     distance_total,
                                     vel_mean_distance,
                                     vel_mean_segment,
                                     vel_top1,
                                     vel_top2,
                                     vel_top3,
                                     acc_top1,
                                     acc_top2,
                                     acc_top3,
                                     vel_low_rate,
                                     vel_change_rate,
                                     bearing_change_rate]], columns=seg_columns)
        
        
            #Add row to segment dataframe
            df_seg = df_seg.append(df_temp, ignore_index=True)

    def save_to_csv(self, file_path):
        """
        Function takes dataframe and save information in a CSV file
        input:  file_path: path to save CSV file
        output: None
        """
        self.df_seg.to_csv(file_path)

# ================================================================
# Segment Featurization
# ================================================================
             
if __name__ == "__main__":
    sf = SegmentFeaturization("segment_master.csv")
    sf.featurize_segment()
    sf.save_to_csv("segment_featured_master.csv")
