import numpy as np
import cv2
import torch
import colorsys
from sklearn.cluster import KMeans
from collections import Counter
from sportsfield_release.utils import utils

class Player:

    def __init__(self,id,isVisible,color=None,team=None,x=None,y=None,positionOnTemplate=None,positionHistory=None):
        self.id = id
        self.isVisible = isVisible
        self.x = x
        self.y = y
        self.team = team
        self.color = color
        self.positionOnTemplate = positionOnTemplate
        self.positionHistory = positionHistory

    def updatePosition(self, x, y):
        self.x = x
        self.y = y
        self.isVisible = True

    def showInfo(self):
        print('id: '+str(self.id)+'  x: '+str(self.x)+'  y: '+str(self.y)+'  xy_dst: '+str(self.positionOnTemplate)+'  team: '+str(self.team)+'  color: '+str(self.color))

    def getWarpedPosition(self):
        return self.warpedPosition
        
    def getColor(self):
        return self.color
    
    def getTeam(self):
        return self.team

    def getPosition(self):
        return self.x, self.y

    def drawPlayerOnPitch(self, img):
        # img[self.y, self.x] = self.color
        cv2.circle(img, (int(self.positionOnTemplate[0]),int(self.positionOnTemplate[1])), 5, (int(self.color[0]), int(self.color[1]), int(self.color[2])), -1)
        # cv2.circle(img, (int(self.positionOnTemplate[0]),int(self.positionOnTemplate[1])), 5, self.color, -1)
        # cv2.putText(img, self.team, (self.x, self.y - 2), 0, 1 / 3, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        self.isVisible = False


    def assignTeam(self, players):
        if self.team is None:
            global main_colors
            global color_team1
            global color_team2
            global color_other          
            # hsv_player = cv2.cvtColor(np.uint8([[self.color]]), cv2.COLOR_BGR2HSV)

            # bgr_max = np.uint8([[main_colors[max_value]]])
            # hsv_max = cv2.cvtColor(bgr_max, cv2.COLOR_BGR2HSV)

            # bgr_med = np.uint8([[main_colors[med_value]]])
            # hsv_med = cv2.cvtColor(bgr_med, cv2.COLOR_BGR2HSV)
            
            # bgr_min = np.uint8([[main_colors[min_value]]])
            # hsv_min = cv2.cvtColor(bgr_min, cv2.COLOR_BGR2HSV)
            # print(self.id, self.color)
            adjust_array = np.array([15, 15, 40])

            # lower_team1 = np.array([color_team1[0]-10, 100, 100])
            # upper_team1 = np.array([color_team1[0]+10, 255, 255])

            # lower_team2 = np.array([color_team2[0]-10, 100, 100])
            # upper_team2 = np.array([color_team2[0]+10, 255, 255])

            # lower_other = np.array([color_other[0]-10, 100, 100])
            # upper_other = np.array([color_other[0]+10, 255, 255])
            lower_team1 = np.subtract(color_team1, adjust_array)
            upper_team1 = np.add(color_team1, adjust_array)

            lower_team2 = np.subtract(color_team2, adjust_array)
            upper_team2 = np.add(color_team2, adjust_array)

            lower_other = np.subtract(color_other, adjust_array)
            upper_other = np.add(color_other, adjust_array)

            mask_team1 = cv2.inRange(self.color, lower_team1, upper_team1)
            mask_team2 = cv2.inRange(self.color, lower_team2, upper_team2)
            mask_other = cv2.inRange(self.color, lower_other, upper_other)
            nonZero1 = cv2.countNonZero(mask_team1)
            nonZero2 = cv2.countNonZero(mask_team2)
            nonZero3 = cv2.countNonZero(mask_other)
            
            maxNonZero = max(nonZero1, nonZero2, nonZero3)
            
            if maxNonZero == nonZero1:
                self.team = "Team_1"
                self.color = cv2.cvtColor(np.uint8([[color_team1]]), cv2.COLOR_HSV2BGR).flatten()
            elif maxNonZero == nonZero2:
                self.team = "Team_2"
                self.color = cv2.cvtColor(np.uint8([[color_team2]]), cv2.COLOR_HSV2BGR).flatten()
            else:
                self.team = "Other"
                self.color = cv2.cvtColor(np.uint8([[color_other]]), cv2.COLOR_HSV2BGR).flatten()

            # print(self.color, self.team)

            # distances = np.sqrt(np.sum((main_colors-self.color)**2,axis=1))

            # index_of_smallest = np.where(distances==np.amin(distances))
            # smallest_distance = main_colors[index_of_smallest]
            # if np.all(smallest_distance == main_colors[max_value]):
            #     self.color = smallest_distance.flatten()
            #     self.team = "Team_1"
            #     # print(self.color, self.team)
            # elif np.all(smallest_distance == main_colors[med_value]):
            #     self.color = smallest_distance.flatten()
            #     self.team = "Team_2"
            #     # print(self.color, self.team)
            # else:
            #     self.color = self.color
            #     self.team = "Other"
            #     # print(self.color, self.team)


    def transformPosition(self, size_in, h, size_out):
        x_norm = normalize(self.x, size_in[1])
        y_norm = normalize(self.y, size_in[0])
        # print("xy_norm: "+str(x_norm)+"___"+str(y_norm))
        normalizedPosition = np.array([[x_norm/2, y_norm/2]], dtype=np.float32)
        # print("normalizedPosition: "+str(normalizedPosition))
        warpedPosition = perspectiveTransform(normalizedPosition, h)
        
        if self.positionHistory is not None:
            prevPosition = self.positionHistory[len(self.positionHistory)-1]
            if warpedPosition[0] - prevPosition[0] > abs(0.01):
                # print(warpedPosition[0], prevPosition[0])
                if warpedPosition[0] > prevPosition[0]:
                    warpedPosition[0] = prevPosition[0]+0.01
                else:
                    warpedPosition[0] = prevPosition[0]-0.01

            if warpedPosition[1] - prevPosition[1] > abs(0.01):
                # print(warpedPosition[1], prevPosition[1])
                if warpedPosition[1] > prevPosition[1]:
                    warpedPosition[1] = prevPosition[1]+0.01
                else:
                    warpedPosition[1] = prevPosition[1]-0.01

            self.positionHistory.append((warpedPosition[0], warpedPosition[1]))     
        else:
            self.positionHistory = []
            self.positionHistory.append((warpedPosition[0], warpedPosition[1]))
        # print("warpedPosition: "+str(warpedPosition))
        x_dst = denormalize(warpedPosition[0], size_out[1])
        y_dst = denormalize(warpedPosition[1], size_out[0])
        # if self.positionHistory is not None:
        #     prevPosition = self.positionHistory[len(self.positionHistory)-1]
        #     if x_dst - prevPosition[0] > abs(0.01):
        #         if x_dst > prevPosition[0]:
        #             x_dst = prevPosition[0]+0.01
        #         else:
        #             x_dst = prevPosition[0]-0.01

        #     if y_dst - prevPosition[1] > abs(0.01):
        #         if y_dst > prevPosition[1]:
        #             y_dst = prevPosition[1]+0.01
        #         else:
        #             y_dst = prevPosition[1]-0.01

        #     self.positionHistory.append((x_dst, y_dst))     
        # else:
        #     self.positionHistory = []
        #     self.positionHistory.append((x_dst, y_dst))
        # print("xy_dst: "+str(x_dst)+"___"+str(y_dst))
        self.positionOnTemplate = (x_dst, y_dst)

once = 0
main_colors = []
color_team1 = []
color_team2 = []
color_other = []
def detectMainColors(players):
    global once
    global main_colors
    global color_team1
    global color_team2
    global color_other
    if once == 0:
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
        # perc = dict(sorted(perc.items()))
        perc = dict(sorted(perc.items(), key=lambda item: item[1]))
        main_colors = clt.cluster_centers_
        max_value = list(perc.keys())[2]
        color_team1 = main_colors[max_value]
        med_value = list(perc.keys())[1]
        color_team2 = main_colors[med_value]
        min_value = list(perc.keys())[0]
        color_other = main_colors[min_value]
        once = 1
        # print(main_colors, perc)
        # return main_colors, max_value, med_value, min_value

def checkIfOnPitch(x, y, h, size):
    x_norm = normalize(x, size[1])
    y_norm = normalize(y, size[0])
    normalizedPosition = np.array([[x_norm/2, y_norm/2]], dtype=np.float32)
    warpedPosition = perspectiveTransform(normalizedPosition, h)
    # print(x, y, warpedPosition)
    return (warpedPosition < 1.05).all() and (warpedPosition > -1.05).all()

def perspectiveTransform(pts, homo_mat):
    pts = utils.to_torch(pts)
    if homo_mat.shape == (3, 3):
        homo_mat = homo_mat[None]
    assert homo_mat.shape[1:] == (3, 3)
    x, y = pts[:, 0], pts[:, 1]
    xy = torch.stack([x, y, torch.ones_like(x)])
    xy_warped = torch.matmul(homo_mat, xy)
    xy_warped, z_warped = xy_warped.split(2, dim=1)
    xy_warped = 2 * xy_warped / (z_warped + 1e-8)
    xy_warped = xy_warped.permute(0, 2, 1)
    xy_warped = xy_warped.detach().numpy()
    return xy_warped[0][0]

def drawAllPlayers(playerList, img):
    for player in playerList:
        if playerList[player].isVisible == True and playerList[player].team != 'Other':
            playerList[player].drawPlayerOnPitch(img)

def getAllPositions(playerList):
    for player in playerList:
        playerList[player].getPosition()

def transformAllPositions(playerList, size_in, h, size_out):
    for player in playerList:
        playerList[player].transformPosition(size_in, h, size_out)

def k_means(img):
    clt = KMeans(n_clusters=4)
    clt = clt.fit(img.reshape(-1, 3))
    n_pixels = len(clt.labels_)
    counter = Counter(clt.labels_)
    perc = {}
    for i in counter:
        perc[i] = np.round(counter[i]/n_pixels, 2)
    # perc = dict(sorted(perc.items()))
    perc = dict(sorted(perc.items(), key=lambda item: item[1]))

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


def normalize(coord, size):
    coord = (coord - size/2)/(size/2)
    return coord

def denormalize(coord, size):
    coord = size/2 + coord*(size/2)
    return coord
    
def detectPlayerColor(img,x1,x2,y1,y2):
    crop = img[y1:y2, x1:x2]
    height, width, channels = crop.shape
    qrt = crop[int(height/6):int(height/2), int(width/5):int(width/1.25)]
    hsv = cv2.cvtColor(qrt,cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, (33, 25, 102), (84, 255, 161))
    ex_green = cv2.bitwise_and(hsv, hsv, mask=mask_green)
    out = hsv-ex_green

    perc, colors = k_means(hsv)
    # max_value = max(perc, key=perc.get)
    # med_temp = list(sorted(perc.values()))[-2]
    # med_value = list(perc.keys())[list(perc.values()).index(med_temp)]
    max_value = list(perc.keys())[3]
    med_value = list(perc.keys())[2]
    # if np.any(np.around(colors[max_value]) <= 0):
    #     return np.around(colors[med_value])
    # else:
    return np.around(colors[max_value])


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

