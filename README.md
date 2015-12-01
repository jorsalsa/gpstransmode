# GPS - TransMode: Predicting transportation modes based on GPS data


## Project Motivation

Critical to many applications, GPS data brings economic benefits. In 2013 alone, GPS technology boosted the US economy by an estimated $68.7 billion USD (http://www.gps.gov/governance/advisory/meetings/2015-06/leveson.pdf). Motivated by the opportunities that this technology brings, GPS-TransMode leverages supervised learning algorithms to identify transportation modes from raw GPS data.

Understanding Transportation Modes, such as walking, biking, driving or taking a bus, can provide mobile applications with contextual information that can be leveraged to enrich the user experience.

This project leverages a dataset and corresponding working papers published by the Microsoft Research Asia GeoLife project (http://research.microsoft.com/en-us/projects/geolife/default.aspx) as described in the "Resources" section below.

## Definitions

For the purpose of this project, we use the following common definitions:

- "GPS data" is saved as a collection of GPS points in log files.
- A "Segment" consists of a sequence of time-based continuous GPS points P1, P2, ...Pn associated with one "Transportation Mode".
- Each "GPS point" Px contains GPS coordinates (Latitude and Longitude) and a Time stamp.
- The total duration of a segment is the difference between the start and end timestamps.
- The total distance of a segment is equal to the sum of distances between each two consecutive GPS points.

## Dataset description

For the GPS-TransMode project we used a dataset containing GPS logs collected by the Microsoft Research Asia GeoLife project. The dataset is publicly available at http://research.microsoft.com/apps/mobile/Download.aspx?p=b16d359d-d164-469e-9fd4-daa38f2b2e13.

The data is organized in folders, one per user. Each user folder contains a "Trajectory" folder storing the GPS logs reported by each user. Each log file is saved in PLT-format. Additionally, there is a "labels" file containing segments labeled with a transportation mode ("driving", "taking a bus", "riding a bike", "walking", etc).


## Data Engineering Process

### Data Preparation

Each entry in a label file represents a time period during which the user utilized a specific "Transportation Mode" (bus, bycicle, etc). This period of time is delimited by "Start Time" and "End Time". In order to associate each label with the corresponding GPS tracking data we have to search inside the trajectory files. Each Trajectory file contains a list of GPS points as measured by the GPS sensor. Each set of GPS points contains the "Latitude", "Longitude", and "Timestamp" of the time of measure.

In order to associate the labeled time period with the GPS tracking points, we first normalize all timestamps (For this particular project We are normalizing datetimes using EPOCH = 1/1/1970). For each time period between "Start Time" and "End Time" at the label file, we collect the corresponding GPS tracking points. This group of points become one segment associated with one Transportation Mode.

### Feature Engineering

Once a segment is defined, we extract critical Physics-based characteristics for each GPS point in the segment according to the latitude, longitude and timestamp information. Critical characteristics considered for this project are: distance between consecutive points, time difference between points, velocity, rate of change of velocity from point to point, acceleration, and changes in direction from previous point.

Based on these characteristics, we finally calculate unique features for each segment, including: time difference between start and end timestamps, total distance, average velocity, top velocities and top acceleration values inside the segment, percentage variation from a specific threshold for changes in direction in the segment, number of points in the segment that velocity felt below a pre-defined threshold  (for this project we use 15% below average human walking speed).

This final set of featurized segments become the input dataset that is used for the modeling phase.

### Modeling

For this project, five different classification algorithms were tested. Goals of this phase were to identify a model that has a high level of overall accuracy and precision, as well as to give us an understanding of what features are important in determining "Transportation Mode".

The classification models are listed below with optimal parameters found via GridSearchCV and tested via K-fold cross-validation:

- Logistic Regression with an inverse-regularization parameter C=1000:
    Accuracy score: 74%
- K-Nearest Neighbors with n=20:
    Accuracy score: 77%
- Single Decision Tree using "Entropy" criterion and a max depth of 10
    Accuracy score: 82%
- SVM using "RBF" kernel function, and Gamma parameter = 0.1
    Accuracy score: 83%
- Random Forest wiht 200 estimators
    Accuracy score: 86%

Based on these results, Random Forest Classifier was selected as the best predictive model.

### Assessing the Confusion matrix

Having selected Random Forest, we proceded to research the confusion matrix of the model. Following encouraging results were obtained, especially due to the high Precision values:

#### Precision:
     - 82% for car: 18% goes to bus, no walk or bike
     - 84% for bus: rest goes mostly to car (8%) while bike and walk take 4% each  
     - 85% for bike: 7% goes to bus and 7% to walk
     - 88% for walk: 5% goes to bike and 5% to bus

#### Recall:
    - 65% for car: rest goes mostly to bus (33%)
    - 84% for bike: rest goes to walk (10%) and bus (5%), no car
    - 85% for bus: rest goes to walk (7%), bike(4%) and car(2%)
    - 92% for walk: rest goes to bike/bus (4% each), no car

## Next steps

### Real-time processing
Initial tests with GPS data not used during the training/test cycle are encouraging. Additional testing will be needed in assessing how fast the model can predict a transportation mode real-time. One of the key areas to further investigate will be finding optimal times between GPS recordings.

###Integration with mobile apps
Integrating a predictive model, such as the one described in this project, in a mobile phone opens intriguing opportunities for the user. We imagine potential mobile applications leveraging the transportation mode prediction in different scenarios:

- improved search results depending on the transportation method (a restaurant search should offer different options whether user is walking or driving),
- optimized trip planning in a foreign city depending whether user is driving or using public transportation,
- safety-aware apps based on location alerting users of "dangerous" areas if walking.

and many more.

## Resources

[1] Yu Zheng, Like Liu, Longhao Wang, Xing Xie. Learning Transportation Modes from Raw GPS Data for Geographic Application on the Web, In Proceedings of International conference on World Wild Web (WWW 2008), Beijing, China. ACM Press: 247-256
[2] Yu Zheng, Quannan Li, Yukun Chen, Xing Xie. Understanding Mobility Based on GPS Data. In Proceedings of ACM conference on Ubiquitous Computing (UbiComp 2008), Seoul, Korea. ACM Press: 312–321.
[3] Yu Zheng, Yukun Chen, Quannan Li, Xing Xie, Wei-Ying Ma. Understanding transportation modes based on GPS data for Web applications. ACM Transaction on the Web. Volume 4, Issue 1, January, 2010. pp. 1-36.