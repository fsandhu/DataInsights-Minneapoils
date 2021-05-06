# CSCE474Project

# requirements.txt

In order to run any of the programs in this project, you will need to have the required libraries installed. In order to facilitate this process we have provided this requirements document, which in conjunction with pip can install all the needed libraries. Use it as such:

```
pip install -r requirements.txt
```

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
