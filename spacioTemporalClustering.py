import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN as dbscan
import datetime as dt
import os

import json
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import seaborn as sns
import argparse
from PIL import Image
import glob
import subprocess


pd.options.mode.chained_assignment = None

#Names of the week days if we want to print out that columns ever
dayNames = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

#Constant to try to equalize distance and time
timeConst = 86400*14*10

class fromDateAction(argparse.Action):

    def __init__(self,option_strings,dest,nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, "fromYear", values.year)
        setattr(namespace, "fromMonth", values.month)
        setattr(namespace, "fromDay", values.day)

class toDateAction(argparse.Action):

    def __init__(self,option_strings,dest,nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, "toYear", values.year)
        setattr(namespace, "toMonth", values.month)
        setattr(namespace, "toDay", values.day)

def toGif(name):
    if not os.path.exists("gifs"):
        os.mkdir("gifs")
    # Create the frames
    frames = []
    imgs = glob.glob(os.path.join("images","*.png"))
    imgs.sort()
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save(os.path.join("gifs","{}.gif".format(name)), format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=1000, loop=0)
    #os.rmtree("images")

def clusteringAnalysis(df,cols,separators,args,
                        xlim=None,ylim=None,eps=0.005,
                        name=None,
                        startDate=dt.datetime(2008,1,1).timestamp()/timeConst,
                        stopDate=dt.datetime(2021,3,14).timestamp()/timeConst):
    #plot = px.scatter_3d(testDataset, x="longitude", y="latitude",z="Datetime",color="cluster")

    for col in separators:
        for i in separators[col]:
            tmpName = "{}_{}={}".format(name,col,i)
            weeklyAnalysis(df.loc[(df[col]==i) & (df.Datetime >= startDate) & (df.Datetime <= stopDate)],
                            cols,tmpName,xlim=xlim,ylim=ylim,eps=eps)
            copyFilePath = os.path.join(os.getcwd(),"images/",name)
            if not os.path.exists(copyFilePath):
                os.mkdir(copyFilePath)
            imageDir = os.path.join(os.getcwd(),"images/")
            subprocess.call("cp -R {}*.png {}/".format(imageDir,copyFilePath),shell=True)
            subprocess.call("rm -r {}/*.png".format(imageDir),shell=True)
    pass

def weeklyAnalysis(df,cols,name,xlim=None,ylim=None,eps=0.005):
    if not os.path.exists("images"):
        os.mkdir("images")
    dayNumMax = df.DayNumber.max()
    dayNumMin = df.DayNumber.min()
    orca = False
    # print("DayNumMin: {}\nDayNumMax: {}\nUnique: ".format(dayNumMin,dayNumMax),)
    # print(df.DayNumber.unique())
    for day in range(dayNumMin,max([dayNumMin+1,dayNumMax-5])):
        testData = df.loc[(df["DayNumber"] >= day) & (df["DayNumber"] < day+7)]
        testData.is_copy = False
        if testData.size == 0:
            continue
        minYear,minMonth,minDay = [testData.Year.min(),testData.Month.min(),testData.Day.loc[testData.Month == testData.Month.min()].min()]
        maxYear,maxMonth,maxDay = [testData.Year.max(),testData.Month.max(),testData.Day.loc[testData.Month == testData.Month.max()].max()]
        fileName = os.path.join("images","{0:s}_{1:d}_{2:02d}_{3:02d}_to_{4:d}_{5:02d}_{6:02d}.png".format(name,minYear,minMonth,minDay,maxYear,maxMonth,maxDay))
        zlim = [None,None]
        zlim = [dt.datetime(minYear,minMonth,minDay).timestamp()/timeConst,
                dt.datetime(maxYear,maxMonth,maxDay).timestamp()/timeConst]
        zlim[1] = max([zlim[1],(dt.datetime.fromtimestamp(zlim[0]*timeConst)+dt.timedelta(6)).timestamp()/timeConst])
        clusters = dbscan(eps=eps).fit(testData[cols])
        numClusters = np.unique(clusters.labels_).size
        testData["cluster"] = clusters.labels_.copy()

        #Visualization
        fig = plt.figure()
        ax = Axes3D(fig)

        #Get the colormap
        cmap = ListedColormap(sns.color_palette("husl",numClusters).as_hex())
        clusterData = testData.loc[testData.cluster >= 0]#Only grab the data that is in a cluster
        #Make sure there is cluster data, otherwise move on
        if clusterData.size == 0:
            print("All noise for {0:d}/{1:02d}/{2:02d} to {3:d}/{4:02d}/{5:02d}\n".format(minYear,minMonth,minDay,maxYear,maxMonth,maxDay))
            clusterData = testData
        sc = ax.scatter(clusterData.longitude,clusterData.latitude,clusterData.Datetime,c=clusterData.cluster,cmap=cmap)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_zlabel("POSIX Datetime/({0:d})".format(timeConst))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        ax.set_zticks(np.arange(zlim[0],zlim[1],(zlim[1]-zlim[0])/7))
        fig.suptitle("{0:d}/{1:02d}/{2:02d} to {3:d}/{4:02d}/{5:02d}".format(minYear,minMonth,minDay,maxYear,maxMonth,maxDay))
        plt.legend(*sc.legend_elements(), bbox_to_anchor=(0, 1), loc=1)
        plt.savefig(fileName)
        plt.close(fig)


    toGif(name)

def getArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--shots", default=False, action="store_true")
    parser.add_argument("--stop" , default=False, action="store_true")
    parser.add_argument("--force", default=False, action="store_true")
    parser.add_argument("-e","--eps",default=0.005, type=float)
    parser.add_argument("-f","--from",default="2008/01/01",type=lambda d: dt.datetime.strptime(d, '%Y/%m/%d'),action=fromDateAction)
    parser.add_argument("-t","--to",default="2021/03/14",type=lambda d: dt.datetime.strptime(d, '%Y/%m/%d'),action=toDateAction)
    parser.add_argument("-n","--name",default="test",type=str)
    parser.add_argument("--typeToValue",type=json.loads)
    args = parser.parse_args()
    return args


def main():
    #Get commandline arguments
    args = getArgs()

    #Define columns for each dataset and a common column set
    cols = ["latitude","longitude","Datetime"]
    columnsStops = ["lat","long","responseDate","problem","reason","race","citationIssued","personSearch","callDisposition"]
    columnsShots = ["latitude","longitude","Response_Date","Problem"]
    columnsForce = ["Y","X","ResponseDate","Problem","Is911Call","PrimaryOffense","SubjectInjury","Race","ForceType","ForceTypeAction"]

    #Create start and stop dates for analysis
    stopDate = dt.datetime(args.toYear,args.toMonth,args.toDay).timestamp()/timeConst
    startDate = dt.datetime(args.fromYear,args.fromMonth,args.fromDay).timestamp()/timeConst

    #If we want to work with the stop data, extract and reformat the data as needed
    if args.stop:
        stopsData = pd.read_csv("./stopDataPreProcessed.csv")
        stopsDataset = stopsData[columnsStops]
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
            datetimes.append(datetime.timestamp()/timeConst)#Change time to an [14] day(hour) metric instead of second

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
        #weeklyAnalysis(stopsDataset,cols,"test")
        xlim = [stopsDataset.longitude.min(),stopsDataset.longitude.max()]
        ylim = [stopsDataset.latitude.min(),stopsDataset.latitude.max()]
        if args.typeToValue is None:
            weeklyAnalysis(stopsDataset.loc[(stopsDataset.Datetime <= stopDate) & (stopsDataset.Datetime >= startDate)],
                        cols,args.name,xlim=xlim,ylim=ylim,eps=args.eps)
        else:
            clusteringAnalysis(stopsDataset,cols,args.typeToValue,args,xlim=xlim,ylim=ylim,eps=args.eps,name=args.name,startDate=startDate,stopDate=stopDate)

    #If we want to work with the force data, extract and reformat the data as needed
    if args.force:
        forceData = pd.read_csv("./forceDataPreProcessed.csv")
        forceDataset = forceData[columnsForce]

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
        for data in dateData:
            times = data.split()
            year,month,day = [int(x) for x in times[0].split('/')]
            time = times[1].split('+')[0]
            weekday = dayNames[dt.datetime(year,month,day).weekday()]
            datetime = dt.datetime(year,month,day,hour,minute,second)
            years.append(year)
            months.append(month)
            days.append(day)
            times_out.append(time)
            weekdays.append(weekday)
            daysIn.append((datetime-minDate).days)
            datetimes.append(datetime.timestamp()/timeConst)#Change time to an hour metric instead of second

        forceDataset["Year"] = years
        forceDataset["Month"] = months
        forceDataset["Day"] = days
        forceDataset["Time"] = times_out
        forceDataset["Datetime"] = datetimes
        forceDataset["DayOfWeek"] = weekdays
        forceDataset["DayNumber"] = daysIn
        forceDataset["latitude"] = forceDataset.pop("Y")#.values[0]
        forceDataset["longitude"] = forceDataset.pop("X")#.values[0]

        xlim = [forceDataset.longitude.min(),forceDataset.longitude.max()]
        ylim = [forceDataset.latitude.min(),forceDataset.latitude.max()]

        if args.typeToValue is None:
            weeklyAnalysis(forceDataset.loc[(forceDataset.Datetime <= stopDate) & (forceDataset.Datetime >= startDate)],
                        cols,args.name,xlim=xlim,ylim=ylim,eps=args.eps)
        else:
            clusteringAnalysis(forceDataset,cols,args.typeToValue,args,xlim=xlim,ylim=ylim,eps=args.eps,name=args.name,startDate=startDate,stopDate=stopDate)


    #If we want to work with the shots data, extract and reformat the data as needed
    if args.shots:
        shotsData = pd.read_csv("./shotsDataPreProcessed.csv")
        shotsDataset = shotsData[columnsShots]
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
        for data in dateData:
            times = data.split()
            year,month,day = [int(x) for x in times[0].split('/')]
            time = times[1].split('+')[0]
            weekday = dayNames[dt.datetime(year,month,day).weekday()]
            datetime = dt.datetime(year,month,day,hour,minute,second)
            years.append(year)
            months.append(month)
            days.append(day)
            times_out.append(time)
            weekdays.append(weekday)
            daysIn.append((datetime-minDate).days)
            datetimes.append(datetime.timestamp()/timeConst)#Change time to an hour metric instead of second

        shotsDataset["Year"] = years
        shotsDataset["Month"] = months
        shotsDataset["Day"] = days
        shotsDataset["Time"] = times_out
        shotsDataset["Datetime"] = datetimes
        shotsDataset["DayOfWeek"] = weekdays
        shotsDataset["DayNumber"] = daysIn

        xlim = [shotsDataset.longitude.min(),shotsDataset.longitude.max()]
        ylim = [shotsDataset.latitude.min(),shotsDataset.latitude.max()]

        if args.typeToValue is None:
            weeklyAnalysis(shotsDataset.loc[(shotsDataset.Datetime <= stopDate) & (shotsDataset.Datetime >= startDate)],
                        cols,args.name,xlim=xlim,ylim=ylim,eps=args.eps)
        else:
            clusteringAnalysis(shotsDataset,cols,args.typeToValue,args,xlim=xlim,ylim=ylim,eps=args.eps,name=args.name,startDate=startDate,stopDate=stopDate)
    pass

if __name__ == "__main__":
    main()
