import random
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

class FCM:
    def __init__(self, data, n_clusters, max_iter, tolerenceRate=1e-6, m=2):
        self.data=data 
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.tolerenceRate = tolerenceRate
        self.centers = None
        self.membershipMatrix = []
        self.lastMembershipMatrix = []

    def calculateDistance(self,a,b):
        i=0
        s=0
        while i < len(a):
            s+=pow(a[i]-b[i],2)
            i+=1
        s=sqrt(s)
        if (s==0.0):
            return 0.001
        return(s)

    def initializeCenters(self):
        i=0
        self.centers=[]
        first_bins= np.linspace(min(self.data[0]),max(self.data[0]),self.n_clusters)
        second_bins= np.linspace(min(self.data[1]),max(self.data[1]),self.n_clusters)
        third_bins= np.linspace(min(self.data[2]),max(self.data[2]),self.n_clusters)
        while i<self.n_clusters:
            newcenter=[first_bins[i],second_bins[i],third_bins[i]]
            self.centers.append(newcenter)
            i+=1
        """ax = plt.axes(projection='3d')
        i=0
        while i<len(self.centers):
            aCenter=self.centers[i]
            xdata=aCenter[0]
            ydata=aCenter[1]
            zdata=aCenter[2]
            ax.scatter3D(xdata, ydata, zdata, c="red")
            i+=1
        plt.show()
        exit()"""
        """while i<self.n_clusters:
            maxrand=len(self.data)
            randomIndex=random.randint(10,maxrand)-11
            row = self.data[randomIndex]
            newcenter=[row[0],row[1],row[2]]
            self.centers.append(newcenter)
            i+=1"""

    def updateMembershipDegrees(self):
        self.lastMembershipMatrix=self.membershipMatrix
        self.membershipMatrix=[]
        for element in self.data:
            distancesOfElement=[]
            for center in self.centers:
                distancesOfElement.append(self.calculateDistance(element,center))
            elementMembership=[]
            for distance in distancesOfElement:
                submem=distance
                s=0
                for d in distancesOfElement:
                    s+=(pow(submem,2)/pow(d,2))
                s=pow(s,1/(self.m-1))
                s=1/s
                elementMembership.append(s)
            self.membershipMatrix.append(elementMembership)

    def updateCenters(self):
        del self.centers[:]
        j=0
        while j< len(self.membershipMatrix[0]):
            ndcolumn=[ndrow[j] for ndrow in self.membershipMatrix]                
            i=0
            l=[]
            while i < len(self.data[0]):
                column=[row[i] for row in self.data]
                s1,s2,s,k = 0,0,0,0
                while k < len(self.data):
                    s1+=column[k]*pow(ndcolumn[k],self.m)
                    s2+=pow(ndcolumn[k],self.m)
                    k+=1   
                s=s1/s2
                l.append(s)    
                i+=1
            j+=1        
            self.centers.append(l)

    def almostSimilar(self,a,b):
        if abs(a-b)<self.tolerenceRate:
            return True
        return False

    def membershipConvergence(self):
        i=0
        j=0
        if len(self.lastMembershipMatrix)>0:
            while (i<len(self.membershipMatrix)):
                while (j<self.n_clusters):
                    if (not self.almostSimilar(self.membershipMatrix[i][j],self.lastMembershipMatrix[i][j])):
                        return False
                    j+=1
                i+=1
            return True
        else:
            return False
    
    def makeClusters(self): #this mothod defuzzify the membership values to make each datapoint belong to only one cluster(kmeans approach)
        clusters=[]
        i=0
        while i < self.n_clusters:
            l=[]
            j=0
            while j< len(self.membershipMatrix):
                if self.membershipMatrix[j][i]==max(self.membershipMatrix[j]):
                    l.append(self.data[j])
                j+=1
            clusters.append(l)
            i+=1
        return clusters

    
    
