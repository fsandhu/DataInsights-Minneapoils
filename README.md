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
