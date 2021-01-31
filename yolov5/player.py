import numpy as np
import cv2
import torch
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

    def updatePosition(self, x, y):
        self.x = x
        self.y = y

    def showInfo(self):
        print('id: '+str(self.id)+'  label: '+str(self.label)+'  team: '+str(self.team)+'  x: '+str(self.x)+'  y: '+str(self.y))

    def assignTeam(self, players):
        if self.team is None:
            temp_list = []
            for key in players:
                temp_list.append(players[key].color)
            color_matrix = np.vstack((temp_list))
            clt = KMeans(n_clusters=3)
            clt.fit(color_matrix)
            n_pixels = len(clt.labels_)
            counter = Counter(clt.labels_)
            perc = {}
            for i in counter:
                perc[i] = np.round(counter[i]/n_pixels, 2)
            perc = dict(sorted(perc.items()))

            main_colors = clt.cluster_centers_

            max_value = max(perc, key=perc.get)
            med_temp = list(sorted(perc.values()))[-2]
            med_value = list(perc.keys())[list(perc.values()).index(med_temp)]
            min_value = min(perc, key=perc.get)

            # hsv_player = cv2.cvtColor(np.uint8([[self.color]]), cv2.COLOR_BGR2HSV)

            # bgr_max = np.uint8([[main_colors[max_value]]])
            # hsv_max = cv2.cvtColor(bgr_max, cv2.COLOR_BGR2HSV)

            # bgr_med = np.uint8([[main_colors[med_value]]])
            # hsv_med = cv2.cvtColor(bgr_med, cv2.COLOR_BGR2HSV)
            
            # bgr_min = np.uint8([[main_colors[min_value]]])
            # hsv_min = cv2.cvtColor(bgr_min, cv2.COLOR_BGR2HSV)

            # adjust_array = np.array([10, 10, 40])

            # lower_team1 = np.subtract(hsv_max, adjust_array)
            # upper_team1 = np.add(hsv_max, adjust_array)

            # lower_team2 = np.subtract(hsv_med, adjust_array)
            # upper_team2 = np.add(hsv_med, adjust_array)

            # lower_team3 = np.subtract(hsv_min, adjust_array)
            # upper_team3 = np.add(hsv_min, adjust_array)

            # mask_team1 = cv2.inRange(hsv_player, lower_team1, upper_team1)
            # mask_team2 = cv2.inRange(hsv_player, lower_team2, upper_team2)
            # mask_team3 = cv2.inRange(hsv_player, lower_team3, upper_team3)
            
            # nonZero1 = cv2.countNonZero(mask_team1)
            # nonZero2 = cv2.countNonZero(mask_team2)
            # nonZero3 = cv2.countNonZero(mask_team3)
            
            # maxNonZero = max(nonZero1, nonZero2, nonZero3)
            
            # if maxNonZero == nonZero1:
            #     self.team = 1
            #     self.color = main_colors[max_value]
            # elif maxNonZero == nonZero2:
            #     self.team = 2
            #     self.color = main_colors[med_value]
            # else:
            #     self.team = 3
            #     self.color = main_colors[min_value]


            distances = np.sqrt(np.sum((main_colors-self.color)**2,axis=1))

            index_of_smallest = np.where(distances==np.amin(distances))
            smallest_distance = main_colors[index_of_smallest]
            if np.all(smallest_distance == main_colors[max_value]):
                self.color = smallest_distance.flatten()
                self.team = "Team_1"
                # print(self.color, self.team)
            elif np.all(smallest_distance == main_colors[med_value]):
                self.color = smallest_distance.flatten()
                self.team = "Team_2"
                # print(self.color, self.team)
            else:
                self.color = self.color
                self.team = "Other"
            # print(type(smallest_distance))
            # print (main_colors[max_value], main_colors[med_value], main_colors[min_value])

def transformPosition(x, y, H):
    pts = np.array([[[x,y]]], np.float32)
    pts_torch = torch.tensor(pts)
    # pts1 = pts.reshape(-1, 1, 2).astype(np.float32)
    # print(pts1)
    # dst = cv2.perspectiveTransform(pts, np.linalg.inv(H))
    dst = cv2.perspectiveTransform(pts, torch.inverse(H))
    return dst[0,0,0], dst[0,0,1]

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

# def detectTeamColor(players):
#     temp_list = []
#     for key in players:
#         temp_list.append(players[key].color)
#     color_matrix = np.vstack((temp_list))
#     clt_2 = KMeans(n_clusters=3)
#     clt_2.fit(color_matrix)
#     main_colors = clt_2.cluster_centers_
    
#     team1 = main_colors[0]
#     team2 = main_colors[1]
#     team3 = main_colors[2]

#     print(team1)

def detectPlayerColor(img,x1,x2,y1,y2):
    crop = img[y1:y2, x1:x2]
    height, width, channels = crop.shape
    qrt = crop[int(height/4):int(height/2), int(width/5):int(width/1.25)]
    
    perc, colors = k_means(qrt)
    max_value = max(perc, key=perc.get)
    return colors[max_value]

def check_color_manual(img,x1,x2,y1,y2):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    crop = hsv[y1:y2, x1:x2]

    height, width, channels = crop.shape
    qrt = crop[int(height/4):int(height/2), int(width/5):int(width/1.25)]

    # rgb
    team1 = np.array([37,69,234])
    team2 = np.array([255,217,215])
    team3 = np.array([0,0,0])

    # hsv
    lower_team1 = np.array([-5, 225, 215])
    upper_team1 = np.array([15, 245, 295])

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

def check_color_manual2(clicks,img,x1,x2,y1,y2):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    crop = hsv[y1:y2, x1:x2]

    height, width, channels = crop.shape
    qrt = crop[int(height/4):int(height/2), int(width/5):int(width/1.25)]

    team1 = np.array(clicks[0])
    team2 = np.array(clicks[1])
    team3 = np.array(clicks[2])

    bgr_A = np.uint8([[clicks[0]]])
    hsv_A = cv2.cvtColor(bgr_A, cv2.COLOR_BGR2HSV)

    bgr_B = np.uint8([[clicks[1]]])
    hsv_B = cv2.cvtColor(bgr_B, cv2.COLOR_BGR2HSV)
            
    bgr_C = np.uint8([[clicks[2]]])
    hsv_C = cv2.cvtColor(bgr_C, cv2.COLOR_BGR2HSV)

    adjust_array = np.array([10, 10, 40])

    lower_team1 = np.subtract(hsv_A, adjust_array)
    upper_team1 = np.add(hsv_A, adjust_array)

    lower_team2 = np.subtract(hsv_B, adjust_array)
    upper_team2 = np.add(hsv_B, adjust_array)

    lower_team3 = np.subtract(hsv_C, adjust_array)
    upper_team3 = np.add(hsv_C, adjust_array)

    mask_team1 = cv2.inRange(qrt, lower_team1, upper_team1)
    mask_team2 = cv2.inRange(qrt, lower_team2, upper_team2)
    mask_team3 = cv2.inRange(qrt, lower_team3, upper_team3)

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
    # print (color)
    return (team, color)

