import numpy as np
import cv2
import colorsys
from sklearn.cluster import KMeans
from collections import Counter

class Player:

    def __init__(self,id,label,color=None,team=None,x=None,y=None):
        self.id = id
        self.x = x
        self.y = y
        self.label = label
        self.team = team
        self.color = color

    def assignTeam(self, players):
        #main_colors = detectTeamColor(players)
        team1 = np.array([37,69,234])
        team2 = np.array([255,217,215])
        # distance1 = np.sqrt(np.sum((team1-self.color)**2))
        # distance2 = np.sqrt(np.sum((team2-self.color)**2))
        # if (distance1 < distance2).all():
        #     self.team = 1
        #     self.color = team1
        # else:
        #     self.team = 2
        #     self.color = team2
        #index_of_smallest = np.where(distances==np.amin(distances))
        #smallest_distance = main_colors[index_of_smallest]
        #self.color = smallest_distance[0]
        # if (smallest_distance == main_colors[0]).all():
        #     self.team = 1
        # else:
        #     self.team = 2
        
        #return smallest_distance
        #orange team
        lower_team1 = np.array([-5, 204, 200])
        upper_team1 = np.array([15, 224, 280])
        #white team
        lower_team2 = np.array([109, 27, 215])
        upper_team2 = np.array([129, 47, 295])

        mask_team1 = cv2.inRange(crop, lower_team1, upper_team1)
        mask_team2 = cv2.inRange(crop, lower_team2, upper_team2)

        sum_team1 = np.sum(mask_team1)
        sum_team2 = np.sum(mask_team2)

        if sum_team1 > sum_team2:
            self.team = 1
            self.color = team1
        else:
            self.team = 2
            self.color = team2


def k_means(img):
    clt = KMeans(n_clusters=4)
    clt = clt.fit(img.reshape(-1, 3))
    n_pixels = len(clt.labels_)
    counter = Counter(clt.labels_)
    perc = {}
    for i in counter:
        perc[i] = np.round(counter[i]/n_pixels, 2)
    perc = dict(sorted(perc.items()))

    return perc, clt.cluster_centers_

def check_color(img,x1,x2,y1,y2):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    crop = hsv[y1:y2, x1:x2]
    #crop = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    height, width, channels = crop.shape
    qrt = crop[int(height/4):int(height/2), int(width/5):int(width/1.25)]
    # perc, colors = k_means(qrt)
    # max_value = max(perc, key=perc.get)
    # return colors[max_value]
    team1 = np.array([37,69,234])
    team2 = np.array([255,217,215])
    team3 = np.array([0,0,0])
    lower_team1 = np.array([-5, 225, 215])
    upper_team1 = np.array([15, 245, 295])
    #white team
    lower_team2 = np.array([108, 33, 215])
    upper_team2 = np.array([128, 53, 295])

    lower_team3 = np.array([144, 25, 11])
    upper_team3 = np.array([164, 45, 91])

    mask_team1 = cv2.inRange(qrt, lower_team1, upper_team1)
    mask_team2 = cv2.inRange(qrt, lower_team2, upper_team2)

    mask_team3 = cv2.inRange(qrt, lower_team3, upper_team3)

    # out1 = cv2.bitwise_and(crop, crop, mask=mask_team1)
    # out2 = cv2.bitwise_and(crop, crop, mask=mask_team2)

    nonZero1 = cv2.countNonZero(mask_team1)
    nonZero2 = cv2.countNonZero(mask_team2)
    nonZero3 = cv2.countNonZero(mask_team3)
    
    maxNonZero = max(nonZero1, nonZero2, nonZero3)
    
    if maxNonZero == nonZero1:
        team = 1
        color = team1
    elif maxNonZero == nonZero2:
        team = 2
        color = team2
    else:
        team = 3
        color = team3

    return (team, color)

def detectTeamColor(players):
    temp_list = []
    for key in players:
        temp_list.append(players[key].color)
    color_matrix = np.vstack((temp_list))
    clt_2 = KMeans(n_clusters=3)
    clt_2.fit(color_matrix)
    return clt_2.cluster_centers_
