# CSCE474Project

# requirements.txt

In order to run any of the programs in this project, you will need to have the required libraries installed. In order to facilitate this process we have provided this requirements document, which in conjunction with pip can install all the needed libraries. Use it as such:

```
pip install -r requirements.txt
```

# 474Project.ipynb

This is the notebook file where we did some genereal preprocessing of the data. We removed attributes that weren't of use to us and replaced null binary values with false or no in this case. We did some preprocessing in Excel and most of that was to remove records that had 0 in latitude and longitude. We also used Excel to split the data into postBLM and preBLM data.

# Weka

We used Weka to generate our association rules. We ran the algorithm on the preprocessed data, there were some errors that have to be solved before you can run them. To remove the errors you need to remove all the apostrophes from the data. We used a minimum confidence of 0.7 for all the runs.

For stop data, first we ran the algorithm on all the attributes but no interesting rules were observed. Then we used the stopDataProcessed.csv file to choose sets of two attributes at a time and then ran the algorithm multiple times to get interesting rules. The dataset was heavily biased in some attributes so we also balanced them all in different csv files and then ran the algorithm again. The rules generated are included as png files in the zip.

Similar analysis was done with the Use Of Force dataset (forceDataAssociation.csv) and the Shots Fired dataset (shotsDataPreProcessed.csv)

Then we also visualized the race attribute in these datasets to see if there was any signifanct differences after the BLM protests (25 May 2020). We only had data for ~10 months after the BLM protests so we took data from only ~10 months before BLM protests to get better results.

# spatioTemporalClustering.py

This is the program that does spatio-temporal clustering analysis, as one would infer from the name. The program has a variety of commandline arguments that allow the user to customize their output. The program will then output a number of visuals to allow the user to interpret them. For the two main modes: static and dynamic, static will output a single png image, while dynamic will output a number of png images and also a gif. These outputs are placed in `./images/static` for static analysis images `./images` for dynamic images and `./gifs` for gifs.

The commandline options are extensive. To get help while running the program simply run the following command:

```
python spatioTemporalClustering.py -h
```

This will list all the commandline options and their help strings. For the most important ones, they will be repeated here.
- Mode flags: `--weekly --monthly --monthlyByWeek --custom --static` are the flags used to tell the program what kind of analysis you want run. Picking anything but static will run dynamic analysis and simply change the time frame analyzed and the steps between the analysis. In this program a month is considered 30 days. These flags are mutually exclusive, only one can be used at a time. The default is `--weekly`.
  * **Note**: If you use `--custom` you will need to provide values for `--customRange` and `--customStep` in order to deviate from the behvior of weekly.
- Static flags: If you do select static as the mode, then you care about how the static analysis is done. The flags for that are `--standard --time --weekday`, where standard jsut does analysis over the date range with no changes, time ignores the year, month and day of the data and weekday does a similar thing but separates data by weekday. These flags are mutually exclusive. The default is standard.
- From date: `-f` or `--from`, provide the date to start the analysis on, inclusive, in the format YYYY-mm-dd
- To date: `-t` or `--to`, provide the date to end the analysis on, inclusive, in the format YYYY-mm-dd
- Dataset flags: `--stop --force --shots` are the flags to signify which datasets you want the analysis to be preformed over. Multiple datasets can be chosen. None of them are on by default. One must be provided for the program to do anything.

# SpartialClusteringAndVisualisation  

Using python tools such as Anaconda we imported the data from data sets. On basis of clustering model we predicted the data. We loaded the data from datasets and accquired different set of attributes from the csv file. 
Based on the four parameters(longitudue, latitude,case Number, Police Force ID) the label is about weather we are going to get a 911 call or not. The Attributes are labelled using label encounter for design model.  

Labels are implemented on each data fitting scenario for every considered region. 

Our design aims to provide a clustering feature to represent the design aspects on the basis of police force data. We have implemented a data cleaning process to initiate the design acquisition based on the Numeric responses of the columns where each set of the data attribute is Case-id, X, Y , Problem and Neighborhood. Each set of the clustering model is represented with the features accepted with the design, and its attributes are clustered with correct response based on the data fitting.
The data labeling is modelled with “label_encoder.fit_transform” utilizing the different string values to corresponding numeric response. For each required column with Dataset_table  are applied to visualize as a cluster feature.
Finally we apply DBSCAN algorithm to initiate the data fitting of the Set of columns and row which are implemented. For case of 1000

To observer different clusters formed we performed dbscan algorithm, then for visualization we plot the crime data using pyplot and scatterplot. 
