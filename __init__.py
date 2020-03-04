from sklearn import preprocessing
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from fcm import FCM
import random


def prepareData(aList):
    aList = aList[1:] #getting rid of the header of the dataset (ie: attribute names)
    i=0
    while (i< len(aList)):
        aList[i]=aList[i][2:] #conserving only 3 variables for a better visualisation later on 
        i+=1
    aList = preprocessing.scale(aList) #normalize variables while conserving the affectiveness of each one on the datapoints
    return (aList)

def plotData(aList,data):
    ax = plt.axes(projection='3d')
    colorSet={"r","b","g","y","black","pink"} #to attribute a color for each cluster
    for sublist in aList:
        i=0
        xdata,ydata,zdata=[],[],[]
        while(i< len(sublist)):
            xdata.append(sublist[i][0])
            ydata.append(sublist[i][1])
            zdata.append(sublist[i][2])
            i+=1 
        color=colorSet.pop()
        ax.scatter3D(xdata, ydata, zdata, c=color)
    ax.set_title('3d Scatter plot')
    plt.show()

def plotCenters(centers):
    ax = plt.axes(projection='3d') 
    x,y,z = [],[],[]   
    for item in centers:
        x.append(item[0])
        y.append(item[1])
        z.append(item[2])
    ax.scatter3D(x,y,z)
    plt.show()

if __name__ == "__main__":
    data=[]
    with open("Mall_Customers.csv","r") as dataFile: #dataset of mall_customers
        for line in dataFile:
            row=line.split(",")
            row[-1]=row[-1][:-2]
            data.append(row)
            line=dataFile.readline
    features=data[0][2:]
    #for row in data:  #uncomment this to see raw dataset
    #    print(row)
    data=prepareData(data)
    x=input("enter number of clusters: ") #this dataset can be effectively devided only into 2 clusters
    fcm=FCM(data,x,100) #create the fcm model
    fcm.initializeCenters() #initialize the centers of clusters
    current=0
    while((current<fcm.max_iter) and(not fcm.membershipConvergence())): #while the membership values not converging
        fcm.updateMembershipDegrees() 
        fcm.updateCenters()      
        current+=1
        """temporaryClusters=fcm.makeClusters()
        for item in temporaryClusters:
            if len(item)==0:
                maxrand=len(data)
                randomIndex=random.randint(10,maxrand)-11
                row = data[randomIndex] 
                item.append(row)"""
        #print(fcm.membershipMatrix[0][0])
        #plotData(temporaryClusters,fcm.data)
        #plotCenters(fcm.centers)

    finalClusters=fcm.makeClusters() #get clusters based on the membership values in the membership matrix(defuzzification)
    print("after "+str(current)+" iterations:")
    num=0
    for cluster in finalClusters:
        num+=1
        print("Cluster "+str(num)+" having "+str(len(cluster))+" datapoints")
    plotData(finalClusters,fcm.data) #visualise each cluster in a color
    #print(fcm.membershipMatrix)
    #plotCenters(fcm.centers)
