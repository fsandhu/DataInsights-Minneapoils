"""
spacioTemporalClustering.py:
    Author: Brennan Rhoadarmer
    CSCE 474 Project
    - A program used to take in commandline arguments and output images/gifs
    of the spaciotemporal clustering analysis done over the selected data
    - Data needs to be in the same directory in the preprocessing format
    - If libraries are needed to install, run:
        'pip install -r requirements.txt'
            or
        'pip3 install -r requirements.txt'
"""

#imports for analysis
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN as dbscan
import datetime as dt
import os
import json

#imports for visualization
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import seaborn as sns
import argparse
from PIL import Image
import glob
import subprocess

#Set to ignore warning about setting columns with a copy.
#We don't care, we don't want/need to change original datasets
pd.options.mode.chained_assignment = None

#Names of the week days if we want to print out that columns ever
dayNames = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

#Constant to try to equalize distance and time, time is set to ~140 days instead of seconds
timeConst = 86400*14*10

#Constant to turn datetime.timestamp()/timeConst to day
toDay = 14*10

class fromDateAction(argparse.Action):
    """
    An argparse action class to allow the easy setting of from Date
    """

    def __init__(self,option_strings,dest,nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, "fromYear", values.year)
        setattr(namespace, "fromMonth", values.month)
        setattr(namespace, "fromDay", values.day)

class toDateAction(argparse.Action):
    """
    An argparse action class to allow easy setting of to Date
    """

    def __init__(self,option_strings,dest,nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, "toYear", values.year)
        setattr(namespace, "toMonth", values.month)
        setattr(namespace, "toDay", values.day)

def toGif(name):
    """
    Function that creates a gif from the images in the images directory.
    The gif is placed in the gif directory
    """
    if not os.path.exists("gifs"):
        os.mkdir("gifs")
    # Create the frames
    frames = []
    imgs = glob.glob(os.path.join("images","{}*.png".format(name)))
    imgs.sort()
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save(os.path.join("gifs","{}.gif".format(name)), format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=1000, loop=0)

def clusteringAnalysis(df,cols,separators,args,
                        xlim=None,ylim=None,eps=0.005,
                        name=None,
                        startDate=dt.datetime(2008,1,1).timestamp()/timeConst,
                        stopDate=dt.datetime(2021,3,14).timestamp()/timeConst):
    """
    Method that allows weekly analysis over the data over certain values for given columns
    """
    #Loop though the keys
    for col in separators:
        #Loop though the values
        for i in separators[col]:
            #Get a new name based on key/value pair and given name
            tmpName = "{}_{}={}".format(name,col,i)
            if not args.static:
                #Do dynamic analysis over constrained data, based on key/value pair
                dynamicAnalysis(df.loc[(df[col]==i) & (df.Datetime >= startDate) & (df.Datetime <= stopDate)],
                                cols,tmpName,args.customRange,args.customStep, xlim=xlim,ylim=ylim,eps=eps)

                #Create directory to copy images to
                copyFilePath = os.path.join(os.getcwd(),"images/",name)
                if not os.path.exists(copyFilePath):
                    os.mkdir(copyFilePath)
                imageDir = os.path.join(os.getcwd(),"images/")
                #Copy images
                subprocess.call("cp -R {}*.png {}/".format(imageDir,copyFilePath),shell=True)
                #Clear images
                #Done so that the next run does not have old/incorrect images in its gif
                subprocess.call("rm -r {}/*.png".format(imageDir),shell=True)
            elif args.static:
                #Do static analysis over the constrained data, based on key/value pair
                staticAnalysis(df.loc[(df[col]==i) & (df.Datetime >= startDate) & (df.Datetime <= stopDate)],
                                cols,tmpName,args, xlim=xlim,ylim=ylim,eps=eps)
                #Create directory to copy images to
                copyFilePath = os.path.join(os.getcwd(),"images/static/",name)
                if not os.path.exists(copyFilePath):
                    os.mkdir(copyFilePath)
                imageDir = os.path.join(os.getcwd(),"images/")
                #Copy images
                subprocess.call("cp -R {}*.png {}/".format(imageDir,copyFilePath),shell=True)
                #Clear images
                #Done so that the next run does not have old/incorrect images in its gif
                subprocess.call("rm -r {}/*.png".format(imageDir),shell=True)

def staticAnalysis(df,cols,name,args,xlim=None,ylim=None,eps=0.005):
    """
    Method to do satic spacio temporal clustering analysis over a given dataset. Generates an image in ./images/static/
    """

    #Create images directory if it does not exist
    if not os.path.exists(os.path.join("images","static")):
        os.mkdir("images/static")
    #Find the max day number and min day number for easier reference later
    dayNumMax = df.DayNumber.max()
    dayNumMin = df.DayNumber.min()

    #Check if we have data, if not don't keep going
    if df.size == 0:
        print("No data for given dates")
        return

    #Getting the maximum and minimum year,month and day for easy reference later
    #Second and third value have .loc in order to make sure we don't fall out of bounds of month,
    #when crossing months and don't have conflicting months if crossing years agressively
    minYear,minMonth,minDay = [df.Year.min(),df.Month.loc[df.Year == df.Year.min()].min(),df.Day.loc[(df.Month == df.Month.min()) & (df.Year == df.Year.min())].min()]
    maxYear,maxMonth,maxDay = [df.Year.max(),df.Month.loc[df.Year == df.Year.max()].max(),df.Day.loc[(df.Month == df.Month.max()) & (df.Year == df.Year.max())].max()]

    #Get min datatime to normalize data for better formatting
    minDatetime = dt.datetime(minYear,minMonth,minDay).timestamp()/timeConst

    if args.standard:
        #Normalize the datetime to the starting date
        df.Datetime = df.Datetime - minDatetime
    elif args.weekday:
        #Set the datetime based on time and day of week
        df.Datetime = df.set_index(["Year","Month","Day","DayOfWeek","Datetime"]).index.map(lambda x: ((x[4]*timeConst-dt.datetime(x[0],x[1],x[2]).timestamp())+x[3]*86400)/timeConst )
    elif args.time:
        #Set the datetimebased on the time only
        df.Datetime = df.set_index(["Year","Month","Day","Datetime"]).index.map(lambda x: (x[4]-dt.datetime(x[0],x[1],x[2]).timestamp())/timeConst)

    #Create the filename for saving the data
    fileName = os.path.join("images","static","{0:s}_{1:d}_{2:02d}_{3:02d}_to_{4:d}_{5:02d}_{6:02d}.png".format(name,minYear,minMonth,minDay,maxYear,maxMonth,maxDay))

    #Craete the limits for the z-axis/time
    zlim = [0,max(dayNumMax-dayNumMin,1)]

    #Cluster analysis with outside library
    clusters = dbscan(eps=eps).fit(df[cols])

    #Get the number of clusters for easy reference later
    numClusters = np.unique(clusters.labels_).size

    #Set the cluster value for all the data
    df["cluster"] = clusters.labels_.copy()

    #Visualization starts here
    fig = plt.figure()
    ax = Axes3D(fig)

    #Get the colormap
    cmap = ListedColormap(sns.color_palette("husl",numClusters).as_hex())
    clusterData = df.loc[df.cluster >= 0]
    #Only grab the data that is in a cluster
    #Make sure there is cluster data, otherwise print that there is not, and use the data with only noise
    if clusterData.size == 0:
        print("All noise for {0:d}/{1:02d}/{2:02d} to {3:d}/{4:02d}/{5:02d}\n".format(minYear,minMonth,minDay,maxYear,maxMonth,maxDay))
        clusterData = df

    #Create the 3D scatter map, coloring points based on their cluster
    sc = None
    if not args.time:
        sc = ax.scatter(clusterData.longitude,clusterData.latitude,clusterData.Datetime*toDay,c=clusterData.cluster,cmap=cmap)
    elif args.time:
        #Convert our timescale to hours
        mult = toDay/24
        sc = ax.scatter(clusterData.longitude,clusterData.latitude,clusterData.Datetime*mult,c=clusterData.cluster,cmap=cmap)

    #Setting axis attributes so it displays nicely
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Day")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    #Set title of the figure, similar to file name
    fig.suptitle("{0:d}/{1:02d}/{2:02d} to {3:d}/{4:02d}/{5:02d}".format(minYear,minMonth,minDay,maxYear,maxMonth,maxDay))

    #Add a legend
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(0, 1), loc=1)
    #Save the figure
    plt.savefig(fileName)
    #Need to manually close the figure
    plt.close(fig)

def dynamicAnalysis(df,cols,name,customRange,customStep,xlim=None,ylim=None,eps=0.005):
    """
    Method that does spatiotemporal analysis a given range and step size over the given dataset. Creates an image plot for each range and also a gif at the end
    """
    #Create images directory if it does not exist
    if not os.path.exists("images"):
        os.mkdir("images")
    #Find the max day number and min day number for easier reference later
    dayNumMax = df.DayNumber.max()
    dayNumMin = df.DayNumber.min()

    #Loop through all of the given days
    #Second value has max so that the loop runs at least once, if only one day is given
    for day in range(dayNumMin,max([dayNumMin+1,dayNumMax-customRange+2]),customStep):
        #Extract the data for this loop
        testData = df.loc[(df["DayNumber"] >= day) & (df["DayNumber"] < day+customRange)]

        #Check if we have data, if not don't keep going
        if testData.size == 0:
            continue

        #Getting the maximum and minimum year,month and day for easy reference later
        #Second and third value have .loc in order to make sure we don't fall out of bounds of month,
        #when crossing months and don't have conflicting months if crossing years agressively
        minYear,minMonth,minDay = [testData.Year.min(),testData.Month.loc[testData.Year == testData.Year.min()].min(),testData.Day.loc[(testData.Month == testData.Month.min()) & (testData.Year == testData.Year.min())].min()]
        maxYear,maxMonth,maxDay = [testData.Year.max(),testData.Month.loc[testData.Year == testData.Year.max()].max(),testData.Day.loc[(testData.Month == testData.Month.max()) & (testData.Year == testData.Year.max())].max()]

        #Get min datatime to normalize data for better formatting
        minDatetime = dt.datetime(minYear,minMonth,minDay).timestamp()/timeConst
        testData.Datetime = testData.Datetime - minDatetime

        #Create the filename for saving the data
        fileName = os.path.join("images","{0:s}_{1:d}_{2:02d}_{3:02d}_to_{4:d}_{5:02d}_{6:02d}.png".format(name,minYear,minMonth,minDay,maxYear,maxMonth,maxDay))

        #Craete the limits for the z-axis/time
        zlim = [None,None]
        zlim = [0,customRange]

        #Need to check that start != end in the limits, so forcibly set it to be customRange-1 days later at least
        #zlim[1] = max([zlim[1],(dt.datetime.fromtimestamp(zlim[0]*timeConst)+dt.timedelta(customRange-1)).timestamp()/timeConst])

        #Cluster analysis with outside library
        clusters = dbscan(eps=eps).fit(testData[cols])

        #Get the number of clusters for easy reference later
        numClusters = np.unique(clusters.labels_).size

        #Set the cluster value for all the data
        testData["cluster"] = clusters.labels_.copy()

        #Visualization starts here
        fig = plt.figure()
        ax = Axes3D(fig)

        #Get the colormap
        cmap = ListedColormap(sns.color_palette("husl",numClusters).as_hex())
        clusterData = testData.loc[testData.cluster >= 0]#Only grab the data that is in a cluster
        #Make sure there is cluster data, otherwise print that there is not, and use the data with only noise
        if clusterData.size == 0:
            print("All noise for {0:d}/{1:02d}/{2:02d} to {3:d}/{4:02d}/{5:02d}\n".format(minYear,minMonth,minDay,maxYear,maxMonth,maxDay))
            clusterData = testData

        #Create the 3D scatter map, coloring points based on their cluster
        sc = ax.scatter(clusterData.longitude,clusterData.latitude,clusterData.Datetime*toDay,c=clusterData.cluster,cmap=cmap)

        #Setting axis attributes so it displays nicely
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_zlabel("Day")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

        #Set title of the figure, similar to file name
        fig.suptitle("{0:d}/{1:02d}/{2:02d} to {3:d}/{4:02d}/{5:02d}".format(minYear,minMonth,minDay,maxYear,maxMonth,maxDay))

        #Add a legend
        plt.legend(*sc.legend_elements(), bbox_to_anchor=(0, 1), loc=1)
        #Save the figure
        plt.savefig(fileName)
        #Need to manually close the figure
        plt.close(fig)
    #Create a gif
    toGif(name)

def getArgs():
    parser = argparse.ArgumentParser("Does cluster analysis over spatiotemporal data")

    analysisType = parser.add_mutually_exclusive_group()
    analysisType.add_argument("--static", default=False, action="store_true", help="Flag to do static analysis over the entire date range")
    analysisType.add_argument("--weekly", default=True,  action="store_true", help="Flag to do dynamic analysis day by day over a week at a time over the entire date range")
    analysisType.add_argument("--monthly", default=False, action="store_true", help="Flag to do dynamic analysis day by day over a month(30 days) at a time over the entire date range")
    analysisType.add_argument("--monthlyByWeek", default=False, action="store_true", help="Flag to do dynamic analysis week by week over a month(30 days) at a time over the entire date range")
    analysisType.add_argument("--custom", default=False, action="store_true", help="Flag to do dynamic analysis over a custom range and step size. Must provide '--customRange' and '--customStep' for full functionality")

    staticAnalysis = parser.add_mutually_exclusive_group()
    staticAnalysis.add_argument("--standard", default=True, action="store_true",help="Flag when doing static analysis to treat all data sequentially")
    staticAnalysis.add_argument("--weekday", default=False, action="store_true", help="Flag when doing static analysis to treat all data as if every weekday was the same (e.g. every saturday happens at the same time)")
    staticAnalysis.add_argument("--time",    default=False, action="store_true", help="Flag when going static analysis to treat all the data as if it was on the same day, with only time mattering")

    parser.add_argument("--shots", default=False, action="store_true", help="Flag to do analysis over shots data")
    parser.add_argument("--stop" , default=False, action="store_true", help="Flag to do analysis over stop data")
    parser.add_argument("--force", default=False, action="store_true", help="Flag to do analysis over force data")
    parser.add_argument("-e","--eps",default=0.005, type=float,help="Float value for cluster point spacing (default: 0.005)")
    parser.add_argument("-f","--from",default="2008/01/01",type=lambda d: dt.datetime.strptime(d, '%Y/%m/%d'),action=fromDateAction,help="Date to start the analysis at in YYYY/mm/dd format")
    parser.add_argument("-t","--to",default="2021/03/14",type=lambda d: dt.datetime.strptime(d, '%Y/%m/%d'),action=toDateAction, help="Date to end the analysis at in YYYY/mm/dd format")
    parser.add_argument("-n","--name",default="test",type=str,help="Name of the saved gif/images")
    parser.add_argument("--typeToValue",type=json.loads,help="JSON Dictionary of columns and values to do analysis over"+
                        " (e.g. '{\"Race\":[\"Black\",\"White\"]}'). The provided columns must be available to all chosen datasets."+
                        "\tPossible columns: | Stops: Problem, Reason, Race, CitationIssued, PersonSearch, CallDisposition. | "+
                        "Shots: Problem. | Force: Problem, Is911Call, PrimaryOffense, SubjectInjury, Race, ForceType, ForceTypeAction. |")

    parser.add_argument("--customRange",type=int, default=7, help="Range in days for dynamic analysis. Only used if flagged as '--custom'")
    parser.add_argument("--customStep", type=int, default=1, help="Step size in days for dynamic analysis. Only used if flagged as '--custom'")
    args = parser.parse_args()
    return args


def main():
    #Get commandline arguments
    args = getArgs()

    #Set up customRange and customStep for given flags
    if args.weekly:
        args.customRange = 7
        args.customStep = 1
    if args.monthly:
        args.customRange = 30
        args.customStep = 1
    if args.monthlyByWeek:
        args.customRange = 30
        args.customStep = 7

    #Define columns for each dataset and a common column set
    cols = ["latitude","longitude","Datetime"]
    columnsStops = ["lat","long","responseDate","problem","reason","race","citationIssued","personSearch","callDisposition"]
    columnsShots = ["latitude","longitude","Response_Date","Problem"]
    columnsForce = ["Y","X","ResponseDate","Problem","Is911Call","PrimaryOffense","SubjectInjury","Race","ForceType","ForceTypeAction"]

    #Create start and stop dates for analysis
    stopDate = dt.datetime(args.toYear,args.toMonth,args.toDay+1).timestamp()/timeConst
    startDate = dt.datetime(args.fromYear,args.fromMonth,args.fromDay).timestamp()/timeConst

    #If we want to work with the stop data, extract and reformat the data as needed
    if args.stop:
        #Read in the data
        stopsData = pd.read_csv("./stopDataPreProcessed.csv")
        stopsDataset = stopsData[columnsStops]

        #Extract the badly formatted date/time data
        dateData = stopsDataset.pop("responseDate")
        minDate = [int(x) for x in dateData.min().split()[0].split('/')]
        minDate = dt.datetime(minDate[0],minDate[1],minDate[2])
        years = []
        months = []
        days = []
        times_out = []
        weekdays = []
        datetimes = []
        daysIn = []

        #Correctly formatting the date/time from the read in data, basically preprocessing
        for data in dateData:
            times = data.split()
            year,month,day = [int(x) for x in times[0].split('/')]
            time = times[1].split('+')[0]
            weekday = dt.datetime(year,month,day).weekday()
            hour,minute,second = [int(x) for x in time.split(":")]
            datetime = dt.datetime(year,month,day,hour,minute,second)
            years.append(year)
            months.append(month)
            days.append(day)
            times_out.append(time)
            weekdays.append(weekday)
            daysIn.append((datetime-minDate).days)
            datetimes.append(datetime.timestamp()/timeConst)

        stopsDataset["Year"] = years
        stopsDataset["Month"] = months
        stopsDataset["Day"] = days
        stopsDataset["Time"] = times_out
        stopsDataset["DayOfWeek"] = weekdays
        stopsDataset["Datetime"] = datetimes
        stopsDataset["DayNumber"] = daysIn
        stopsDataset["latitude"] = stopsDataset.pop("lat")#.values[0]
        stopsDataset["longitude"] = stopsDataset.pop("long")#.values[0]
        stopsDataset["Problem"] = stopsDataset.pop("problem")#.values[0]
        stopsDataset.rename(columns = {"reason":"Reason",
                                       "race":"Race",
                                       "citationIssued":"CitationIssued",
                                       "personSearch":"PersonSearch",
                                       "callDisposition":"CallDisposition"}
                            ,inplace=True)

        #Create limits so that the created graphs don't change limits, making them easier to compare
        xlim = [stopsDataset.longitude.min(),stopsDataset.longitude.max()]
        ylim = [stopsDataset.latitude.min(),stopsDataset.latitude.max()]

        #Decide how to do the analysis
        if args.typeToValue is not None:
            clusteringAnalysis(stopsDataset,cols,args.typeToValue,args,xlim=xlim,ylim=ylim,eps=args.eps,name=args.name,startDate=startDate,stopDate=stopDate)
        elif not args.static:
            dynamicAnalysis(stopsDataset.loc[(stopsDataset.Datetime <= stopDate) & (stopsDataset.Datetime >= startDate)],
                        cols,args.name+"_stops",args.customRange, args.customStep, xlim=xlim,ylim=ylim,eps=args.eps)
        else:
            staticAnalysis(stopsDataset.loc[(stopsDataset.Datetime <= stopDate) & (stopsDataset.Datetime >= startDate)],
                        cols,args.name+"_stops",args,xlim=xlim,ylim=ylim,eps=args.eps)

    #If we want to work with the force data, extract and reformat the data as needed
    if args.force:
        #Read in the data
        forceData = pd.read_csv("./forceDataPreProcessed.csv")
        forceDataset = forceData[columnsForce]

        #Extrat the incorrectly formatted date/time data
        dateData = forceDataset.pop("ResponseDate")
        minDate = [int(x) for x in dateData.min().split()[0].split('/')]
        minDate = dt.datetime(minDate[0],minDate[1],minDate[2])
        daysIn = []
        years = []
        months = []
        days = []
        times_out = []
        weekdays = []
        datetimes = []

        #Correctly formatting the date/time from the read in data, basically preprocessing
        for data in dateData:
            times = data.split()
            year,month,day = [int(x) for x in times[0].split('/')]
            time = times[1].split('+')[0]
            weekday = dt.datetime(year,month,day).weekday()
            hour,minute,second = [int(x) for x in time.split(":")]
            datetime = dt.datetime(year,month,day,hour,minute,second)
            years.append(year)
            months.append(month)
            days.append(day)
            times_out.append(time)
            weekdays.append(weekday)
            daysIn.append((datetime-minDate).days)
            datetimes.append(datetime.timestamp()/timeConst)#Change time to an hour metric instead of second

        #Set the correct columns of the dataset for the new formatted date/time
        forceDataset["Year"] = years
        forceDataset["Month"] = months
        forceDataset["Day"] = days
        forceDataset["Time"] = times_out
        forceDataset["Datetime"] = datetimes
        forceDataset["DayOfWeek"] = weekdays
        forceDataset["DayNumber"] = daysIn
        forceDataset["latitude"] = forceDataset.pop("Y")#.values[0]
        forceDataset["longitude"] = forceDataset.pop("X")#.values[0]

        #Get the limits of the dataset so that the created plots don't differ in limits
        xlim = [forceDataset.longitude.min(),forceDataset.longitude.max()]
        ylim = [forceDataset.latitude.min(),forceDataset.latitude.max()]

        #Decide how to do the analysis
        if args.typeToValue is not None:
            clusteringAnalysis(forceDataset,cols,args.typeToValue,args,xlim=xlim,ylim=ylim,eps=args.eps,name=args.name,startDate=startDate,stopDate=stopDate)
        elif not args.static:
            dynamicAnalysis(forceDataset.loc[(forceDataset.Datetime <= stopDate) & (forceDataset.Datetime >= startDate)],
                        cols,args.name+"_force",args.customRange, args.customStep, xlim=xlim,ylim=ylim,eps=args.eps)
        else:
            staticAnalysis(forceDataset.loc[(forceDataset.Datetime <= stopDate) & (forceDataset.Datetime >= startDate)],
                        cols,args.name+"_force",args,xlim=xlim,ylim=ylim,eps=args.eps)


    #If we want to work with the shots data, extract and reformat the data as needed
    if args.shots:
        #Read in the data
        shotsData = pd.read_csv("./shotsDataPreProcessed.csv")
        shotsDataset = shotsData[columnsShots]

        #Extract the incorrectly formatted date/time data
        dateData = shotsDataset.pop("Response_Date")
        minDate = [int(x) for x in dateData.min().split()[0].split('/')]
        minDate = dt.datetime(minDate[0],minDate[1],minDate[2])
        daysIn = []
        years = []
        months = []
        days = []
        times_out = []
        weekdays = []
        datetimes = []

        #Correctly formatting the date/time from the read in data, basically preprocessing
        for data in dateData:
            times = data.split()
            year,month,day = [int(x) for x in times[0].split('/')]
            time = times[1].split('+')[0]
            weekday = dt.datetime(year,month,day).weekday()
            hour,minute,second = [int(x) for x in time.split(":")]
            datetime = dt.datetime(year,month,day,hour,minute,second)
            years.append(year)
            months.append(month)
            days.append(day)
            times_out.append(time)
            weekdays.append(weekday)
            daysIn.append((datetime-minDate).days)
            datetimes.append(datetime.timestamp()/timeConst)#Change time to an hour metric instead of second

        #Set the correct columns in the dataset to the newly formatted data
        shotsDataset["Year"] = years
        shotsDataset["Month"] = months
        shotsDataset["Day"] = days
        shotsDataset["Time"] = times_out
        shotsDataset["Datetime"] = datetimes
        shotsDataset["DayOfWeek"] = weekdays
        shotsDataset["DayNumber"] = daysIn

        #Get the limits of the dataset so that the created plots don't differ in limits
        xlim = [shotsDataset.longitude.min(),shotsDataset.longitude.max()]
        ylim = [shotsDataset.latitude.min(),shotsDataset.latitude.max()]

        #Decide how the analysis will happen
        if args.typeToValue is not None:
            clusteringAnalysis(shotsDataset,cols,args.typeToValue,args,xlim=xlim,ylim=ylim,eps=args.eps,name=args.name,startDate=startDate,stopDate=stopDate)
        elif not args.static:
            dynamicAnalysis(shotsDataset.loc[(shotsDataset.Datetime <= stopDate) & (shotsDataset.Datetime >= startDate)],
                        cols,args.name+"_shots",args.customRange,args.customStep,xlim=xlim,ylim=ylim,eps=args.eps)
        else:
            staticAnalysis(shotsDataset.loc[(shotsDataset.Datetime <= stopDate) & (shotsDataset.Datetime >= startDate)],
                        cols,args.name+"_shots",args,xlim=xlim,ylim=ylim,eps=args.eps)


if __name__ == "__main__":
    main()
