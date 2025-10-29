# Semesterarbeit
## SA: a resilience based approach for taxi travel demand prediction under disruptive events

### Data source: 
    https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew/data

### City office news of Chicago during the COVID-19: 
    https://www.chicago.gov/city/en/sites/covid-19/home/press-releases.html

###   1. Original Data analysis

####  1.1 key milestones in the spread of the COVID-19 pandemic in Chicago

      ->  January 24, 2020: City of Chicago Announces First Local Patient with Travel-Related Case of 2019 Novel Coronavirus
      ->  March 6, 2020: Public Health and Chicago Public Schools Officials Announce New Presumptive Positive Case of Coronavirus Disease 2019
      ->  March 9, 2020: Public Health Officials Report Four Additional COVID-19 Cases in Illinois 
      ->  March 13, 2020: City of Chicago Prepares for Closure of all K-12 Schools, as Mandated by the State of Illinois
      ->  March 17, 2020: Public Health Officials Announce First Illinois Coronavirus Disease Death
      ->  March 19, 2020: City of Chicago Orders Sick Residents to Remain Home to Prevent Further Spread of COVID-19
      ->  March 20, 2020: Mayor Lightfoot Joins Governor Pritzker To Announce State Order to Stay at Home to Prevent Further Spread of COVID-19
      ->  March 26, 2020: Mayor Lightfoot Orders the Immediate Closure of The City's Lakefront, Adjacent Parks, 606 and Riverwalk to the Public
      ->  March 28, 2020: Public Health Officials Announce the First Death of an Infant with Coronavirus Disease
      ->  May 28, 2020: Mayor Lightfoot and CDPH Announce Chicago Ready to Begin Reopening Cautiously on Wednesday, June 3, 2020
      ->  June 26, 2020: Restaurants, bars and breweries serve patrons indoors with limited capacity and safety restrictions
      ->  October 19, 2020: Mayor Lightfoot and CDPH Commissioner Dr. Arwady Sound the Alarm on Second Wave of COVID-19
      ->  October 22, 2020: Mayor Lightfoot, CDPH, and BACP Reinstate Targeted COVID-19 Restrictions in Response to Rapid Rise in Cases and Hospitalizations
      ->  February 10, 2021: Mayor Lightfoot Announces Roadmap for Further Easing of COVID-19 Regulations
      ->  April 29, 2021: Mayor Lightfoot Announces the Launch of “Open Chicago”
      ->  December 21, 2021: City of Chicago Announces Vaccine Requirements for Restaurants, Bars, Gyms, and Other Indoor Public Places
      ->  

#### 1.2 Diagram about the OD demand of City Chicago
![image](https://github.com/EisenHanhan/semesterarbeit/blob/main/IMG/OD_demand_2020.png)
![image](https://github.com/EisenHanhan/semesterarbeit/blob/main/IMG/OD_demand_2019.png)
![image](https://github.com/EisenHanhan/semesterarbeit/blob/main/IMG/OD_demand_Recovery.png)
![image](https://github.com/EisenHanhan/semesterarbeit/blob/main/IMG/OD_demand_early.png)
![image](https://github.com/EisenHanhan/semesterarbeit/blob/main/IMG/Event_recovery_process.png)
 
###   2. Clustering

####  2.1 Methods

#####  2.1.1 Dynamic Time Warping, DTW
![image](https://github.com/EisenHanhan/semesterarbeit/blob/main/IMG/Dynamic%20Time%20Warping.png)
![image](https://github.com/EisenHanhan/semesterarbeit/blob/main/IMG/DTW.png)
#####  2.1.2 LB Keogh Distance
LB_Keogh is tool for lower bounding various time series distance measures. It was introduced in 2002 as the first non trivial lower bound for Dynamic Time Warping (DTW), and it is still the fastest known technique for indexing DTW.
![image](https://github.com/EisenHanhan/semesterarbeit/blob/main/IMG/LB_Keogh.png)
