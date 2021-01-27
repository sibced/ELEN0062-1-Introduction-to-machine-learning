# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 11:30:36 2020

@author: jmore
"""

# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import time
import datetime
from contextlib import contextmanager

import pandas as pd
import pandas_profiling
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

@contextmanager
def measure_time(label):
    """
    Context manager to measure time of computation.
    >>> with measure_time('Heavy computation'):
    >>>     do_heavy_computation()
    'Duration of [Heavy computation]: 0:04:07.765971'

    Parameters
    ----------
    label: str
        The label by which the computation will be referred
    """
    start = time.time()
    yield
    end = time.time()
    print('Duration of [{}]: {}'.format(label,
                                        datetime.timedelta(seconds=end-start)))


def load_from_csv(path, delimiter=','):
    """
    Load csv file and return a NumPy array of its data

    Parameters
    ----------
    path: str
        The path to the csv file to load
    delimiter: str (default: ',')
        The csv field delimiter

    Return
    ------
    D: array
        The NumPy array of the data contained in the file
    """
    return pd.read_csv(path, delimiter=delimiter)

def same_team_(sender,player_j):
    if sender <= 11:
        return int(player_j <= 11)
    else:
        return int(player_j > 11)

def make_pair_of_players(X_, y_=None):
    n_ = X_.shape[0]
    pair_feature_col = ["sender", "x_sender", "y_sender", "player_j", "x_j", "y_j", "same_team"]
    X_pairs = pd.DataFrame(data=np.zeros((n_*22,len(pair_feature_col))), columns=pair_feature_col)
    y_pairs = pd.DataFrame(data=np.zeros((n_*22, 1)), columns=["pass"])

    # From pass to pair of players
    idx = 0
    for i in range(n_):
        sender = X_.iloc[i].sender
        players = np.arange(1, 23)
        #other_players = np.delete(players, sender-1)
        p_i_ = X_.iloc[i]
        for player_j in players:

            X_pairs.iloc[idx] = [sender,  p_i_["x_{:0.0f}".format(sender)], p_i_["y_{:0.0f}".format(sender)],
                                 player_j, p_i_["x_{:0.0f}".format(player_j)], p_i_["y_{:0.0f}".format(player_j)], same_team_(sender, player_j)]

            if not y_ is None:
                y_pairs.iloc[idx]["pass"] = int(player_j == y_.iloc[i])
            idx += 1 
    
    return X_pairs, y_pairs
 
def compute_distance_(X_):
    d = np.zeros((X_.shape[0],))

    d = np.sqrt((X_["x_sender"]-X_["x_j"])**2 + (X_["y_sender"]-X_["y_j"])**2)
    return d

def forward_distance(X_):
    d = np.zeros((X_.shape[0],))

    d = X_["side_sender"]*(X_["x_sender"]-X_["x_j"])
    return d

def lateral_distance(X_):
    d = np.zeros((X_.shape[0],))

    d = abs(X_["y_sender"]-X_["y_j"])
    return d

def opponent_distance_sender(X_):
    """
    Return shortest distance between sender and an opponent
    """
    d = np.zeros((X_.shape[0],))
    n_ = X_.shape[0]
    
    for i in range(0,n_,22):
        if X_.iloc[i].sender > 11:
            di = min(X_[i+0:i+10]["distance"])
        else:
            di = min(X_[i+11:i+21]["distance"])
        for j in range(22):
            d[i+j] = di
    return d


def opponent_distance_receiver(X_):
    """
    Return shortest distance between potential receiver and sender's opponent
    """
    
    d = np.zeros((X_.shape[0],))
    n_ = X_.shape[0]
    
    for i in range(0,n_,22):
        if X_.iloc[i].sender <= 11:
            for j in range(0,11,1):
                dj = np.zeros((11,))
                xj = X_.iloc[i+j].x_j
                yj = X_.iloc[i+j].y_j
                for k in range(11,22,1):
                    dj[k-11] = np.sqrt((xj-X_.iloc[i+k].x_j)**2 + (yj-X_.iloc[i+k].y_j)**2)
                d[i+j] = min(dj)
            for j in range(11,22,1):
                d[i+j] = 0
        else:
            for j in range(0,11,1):
                d[i+j] = 0
            for j in range(11,22,1):
                dj = np.zeros((11,))
                xj = X_.iloc[i+j].x_j
                yj = X_.iloc[i+j].y_j
                for k in range(0,11,1):
                    dj[k] = np.sqrt((xj-X_.iloc[i+k].x_j)**2 + (yj-X_.iloc[i+k].y_j)**2)
                d[i+j] = min(dj)
    return d

def intercept_distance(X_):
    """
    Return shortest distance between segment line between sender and potential receiver
    and a sender's opponent. Teka intot account only the players located in the rectangle
    defined by sender and potential receiver.
    """
    d = np.zeros((X_.shape[0],))
    n_ = X_.shape[0]
    
    for i in range(0,n_-21,22):
        for k in range(0,22):
            A = X_.iloc[i+k].y_sender-X_.iloc[i+k].y_j
            B = X_.iloc[i+k].x_sender-X_.iloc[i+k].x_j
            
            if A == 0 and B==0:
                d[i+k] = 0
            else:
                C = X_.iloc[i+k].x_j*X_.iloc[i+k].y_sender - X_.iloc[i+k].x_sender *X_.iloc[i+k].y_j
                dj = np.zeros((11,))
                if X_.iloc[i+k].player_j <= 11:
                    for j in range(11,22,1):
                        if (X_.iloc[i+k].x_sender < X_.iloc[i+j].x_j < X_.iloc[i+k].x_j or X_.iloc[i+k].x_sender > X_.iloc[i+j].x_j > X_.iloc[i+k].x_j)\
                            and (X_.iloc[i+k].y_sender < X_.iloc[i+j].y_j < X_.iloc[i+k].y_j or X_.iloc[i+k].y_sender > X_.iloc[i+j].y_j > X_.iloc[i+k].y_j):
                                dj[j-11] = abs(A*X_.iloc[i+j].x_j + B*X_.iloc[i+j].y_j + C)/np.sqrt(A*A + B*B)
                        else:
                            dj[j-11] = 5000
                
                else:
                    for j in range(0,11,1):
                        if (X_.iloc[i+k].x_sender < X_.iloc[i+j].x_j < X_.iloc[i+k].x_j or X_.iloc[i+k].x_sender > X_.iloc[i+j].x_j > X_.iloc[i+k].x_j)\
                            and (X_.iloc[i+k].y_sender < X_.iloc[i+j].y_j < X_.iloc[i+k].y_j or X_.iloc[i+k].y_sender > X_.iloc[i+j].y_j > X_.iloc[i+k].y_j):
                                dj[j] = abs(A*X_.iloc[i+j].x_j + B*X_.iloc[i+j].y_j + C)/np.sqrt(A*A + B*B)
                        else:
                            dj[j-11] = 5000 
                            
                d[i+k] = min(dj)           
                if X_.iloc[i+k].same_team == 0:
                        d[i+k] = -d[i+k]
                
    return d

def teams_parameters(X_):
    """
    Return:
    m_s = mean position of sender's team along x-axis
    m_s = mean position of the team of the sender's opponent along x-axis
    gk_n : team of the goalkeeper on the negative side
    gk_p : team of the goalkeeper on the positive side
    """
    m_s = np.zeros((X_.shape[0],))
    m_o = np.zeros((X_.shape[0],))
    gk_p = np.zeros((X_.shape[0],))
    gk_n = np.zeros((X_.shape[0],))
    n_ = X_.shape[0]
    for i in range(0,n_,22):
        x_s = np.zeros((11,))
        x_o = np.zeros((11,))
        d_s_1 = np.zeros((11,))
        d_s_2 = np.zeros((11,))
        d_o_1 = np.zeros((11,))
        d_o_2 = np.zeros((11,))
        if X_.iloc[i].sender <= 11:
            for j in range(0,11,1):
                x_s[j] = X_.iloc[i+j].x_j
                d_s_1[j] = np.sqrt((X_.iloc[i].x_j-(-5250))**2 + (X_.iloc[i].x_j-0)**2)
                d_s_2[j] = np.sqrt((X_.iloc[i].x_j-5250)**2 + (X_.iloc[i].x_j-0)**2)
            for j in range (11,22,1):
                x_o[j-11] = X_.iloc[i+j].x_j
           
        else:
            for j in range(11,22,1):
                
                x_s[j-11] = X_.iloc[i+j].x_j
            for j in range(0,11,1):
                
                x_o[j] = X_.iloc[i+j].x_j
        
        for k in range(22):
            m_s[i+k] = x_s.mean()
            m_o[i+k] = x_o.mean()
            
            if min(d_s_1) < min(d_o_1):
                gk_n[i+k] = 1
            else:
                gk_p[i+k] = 2
            if min(d_s_2) < min(d_o_2):
                gk_p[i+k] = 1
            else:
                gk_p[i+k] = 2
            
    return m_s,m_o,gk_n,gk_p

def side_sender(X_):
    """
    Return the side of the sender's team :
        -1 if the side is on the negative x-axis
        +1 if the side is on the positive x-axis
    """
    side_sender = np.zeros((X_.shape[0],))
    n_ = X_.shape[0]
    for i in range (0,n_,22):
        if (X_.iloc[i].mean_team_s < 0) and (X_.iloc[i].mean_team_o < 0):
            if X_.iloc[i].gk_p == 1:
                side_s = 1
            else:
                side_s = -1
        elif (X_.iloc[i].mean_team_s > 0) and (X_.iloc[i].mean_team_o > 0):
            if X_.iloc[i].gk_n == 1:
                side_s = -1
            else:
                side_s = 1
        else:
            if (X_.iloc[i].mean_team_s < 0):
                side_s = -1
            else:
                side_s = 1
        for k in range(22):
            side_sender[i+k] = side_s
    
    return side_sender

def find_zone(x,y):
    """

    Parameters
    ----------
    x : c coordinate
    y : y coordinate

    Returns
    -------
    z_x : x-zone :
        -2 below -3150
        -1 below -1050
        0 below 1050
        1 below 3150
        2 above 3150
        
    z_y : y-zone :
        -1 below -3400/3
        0 below 3400/3
        1 above 3400/3
    """
    if y < -3400/3:
        if x < -3150:
            z_x = -2
            z_y = -1
        elif x < -1050:
            z_x = -1
            z_y = -1
        elif x < 1050:
            z_x = 0
            z_y = -1
        elif x < 3150:
            z_x = 1
            z_y = -1
        else:
            z_x = 2
            z_y = -1
    elif y < 3400/3:
        if x < -3150:
            z_x = -2
            z_y = 0
        elif x < -1050:
            z_x = -1
            z_y = 0
        elif x < 1050:
            z_x = 0
            z_y = 0
        elif x < 3150:
            z_x = 1
            z_y = 0
        else:
            z_x = 2
            z_y = 0
    else:
        if x < -3150:
            z_x = -2
            z_y = 1
        elif x < -1050:
            z_x = -1
            z_y = 1
        elif x < 1050:
            z_x = 0
            z_y = 1
        elif x < 3150:
            z_x = 1
            z_y = 1
        else:
            z_x = 2
            z_y = 1
            
    return z_x,z_y

def zones(X_):
    """
    Return x and y zones of sender and potential receiver

    """
    n_ = X_.shape[0]
    zx_sender = np.zeros((X_.shape[0],))
    zx_j = np.zeros((X_.shape[0],))
    zy_sender = np.zeros((X_.shape[0],))
    zy_j = np.zeros((X_.shape[0],))
    
    for i in range(0,n_,22):
        #print("main_i"+str(i))
        x_sender = X_.iloc[i].x_sender
        y_sender = X_.iloc[i].y_sender
        zone_x_sender,zone_y_sender = find_zone(x_sender,y_sender)
        zone_x_sender = -1 * zone_x_sender * X_.iloc[i].side_sender
        zone_y_sender = -1 * zone_y_sender * X_.iloc[i].side_sender
        
        if X_.iloc[i].sender < 11:
            coeff = 1
        else:
            coeff = -1
        for j in range(0,11,1):
            x_j = X_.iloc[i + j].x_j
            y_j = X_.iloc[i + j].y_j
            zone_x_j,zone_y_j = find_zone(x_j,y_j)
            zx_j[i+j] = -1 * zone_x_j * X_.iloc[i].side_sender * coeff
            zy_j[i+j] = -1 * zone_y_j * X_.iloc[i].side_sender * coeff
            zx_sender[i + j] = zone_x_sender
            zy_sender[i + j] = zone_y_sender
        for j in range(11,22,1):
            x_j = X_.iloc[i + j].x_j
            y_j = X_.iloc[i + j].y_j
            zone_x_j,zone_y_j = find_zone(x_j,y_j)
            zx_j[i+j] = zone_x_j * X_.iloc[i].side_sender * coeff
            zy_j[i+j] = zone_y_j * X_.iloc[i].side_sender * coeff
            zx_sender[i + j] = zone_x_sender
            zy_sender[i + j] = zone_y_sender
    return zx_sender,zy_sender,zx_j,zy_j

def second_max(list1):
    mx=max(list1[0],list1[1]) 
    secondmax=min(list1[0],list1[1]) 
    n =len(list1)
    for i in range(2,n): 
        if list1[i]>mx: 
            secondmax=mx
            mx=list1[i] 
        elif list1[i]>secondmax and \
            mx != list1[i]: 
            secondmax=list1[i]
    return secondmax

def write_submission(predictions=None, probas=None, estimated_score=0, file_name="submission", date=True, indexes=None):
    """
    Write a submission file for the Kaggle platform

    Parameters
    ----------
    predictions: array [n_predictions, 1]
        `predictions[i]` is the prediction for player 
        receiving pass `i` (or indexes[i] if given).
    probas: array [n_predictions, 22]
        `probas[i,j]` is the probability that player `j` receives
        the ball with pass `i`.
    estimated_score: float [1]
        The estimated accuracy of predictions. 
    file_name: str or None (default: 'submission')
        The path to the submission file to create (or override). If none is
        provided, a default one will be used. Also note that the file extension
        (.txt) will be appended to the file.
    date: boolean (default: True)
        Whether to append the date in the file name

    Return
    ------
    file_name: path
        The final path to the submission file
    """   

    if date: 
        file_name = '{}_{}'.format(file_name, time.strftime('%d-%m-%Y_%Hh%M'))

    file_name = '{}.txt'.format(file_name)

    if predictions is None and probas is None:
        raise ValueError('Predictions and/or probas should be provided.')

    n_samples = 3000
    if indexes is None:
        indexes = np.arange(n_samples)

    if probas is None:
        print('Deriving probabilities from predictions.')
        probas = np.zeros((n_samples,22))
        for i in range(n_samples):
            probas[i, predictions[i]-1] = 1

    if predictions is None:
        print('Deriving predictions from probabilities')
        predictions = np.zeros((n_samples, ))
        for i in range(n_samples):
            mask = probas[i] == np.max(probas[i])
            selected_players = np.arange(1,23)[mask]
            predictions[i] = int(selected_players[0])


    # Writing into the file
    with open(file_name, 'w') as handle:
        # Creating header
        header = '"Id","Predicted",'
        for j in range(1,23):
            header = header + '"P_{:0.0f}",'.format(j)
        handle.write(header[:-1]+"\n")

        # Adding your estimated score
        first_line = '"Estimation",{},'.format(estimated_score)
        for j in range(1,23):
            first_line = first_line + '0,'
        handle.write(first_line[:-1]+"\n")

        # Adding your predictions
        for i in range(n_samples):
            line = "{},{:0.0f},".format(indexes[i], predictions[i])
            pj = probas[i, :]
            for j in range(22):
                line = line + '{},'.format(pj[j])
            handle.write(line[:-1]+"\n")

    return file_name
    
    # %% ---------------------------------- Data loading ------------------------------------- #
if __name__ == '__main__':
    prefix = 'D:/MyDocuments/Master_meca/Machine_learning/Projet_3/iml2020/'

    # Load training data
    X_LS = load_from_csv(prefix+'input_training_set.csv')
    y_LS = load_from_csv(prefix+'output_training_set.csv')
      # Load test data
    X_TS = load_from_csv(prefix+'input_test_set.csv')
    
    # Looking for odd values
    x_bound = 5500
    y_bound = 3550
    indexNames = X_LS[(abs(X_LS['x_1']) > x_bound)|(abs(X_LS['y_1']) > y_bound)|(abs(X_LS['x_2']) > x_bound)|(abs(X_LS['y_2']) > y_bound)|(abs(X_LS['x_3']) > x_bound)|(abs(X_LS['y_3']) > y_bound)|(abs(X_LS['x_4']) > x_bound)|(abs(X_LS['y_4']) > y_bound)|(abs(X_LS['x_5']) > x_bound)|(abs(X_LS['y_5']) > y_bound)|(abs(X_LS['x_6']) > x_bound)|(abs(X_LS['y_6']) > y_bound)|(abs(X_LS['x_7']) > x_bound)|(abs(X_LS['y_7']) > y_bound)|(abs(X_LS['x_8']) > x_bound)|(abs(X_LS['y_8']) > y_bound)|(abs(X_LS['x_9']) > x_bound)|(abs(X_LS['y_9']) > y_bound)|(abs(X_LS['x_10']) > x_bound)|(abs(X_LS['y_10']) > y_bound)|(abs(X_LS['x_11']) > x_bound)|(abs(X_LS['y_11']) > y_bound)|(abs(X_LS['x_12']) > x_bound)|(abs(X_LS['y_12']) > y_bound)|(abs(X_LS['x_13']) > x_bound)|(abs(X_LS['y_13']) > y_bound)|(abs(X_LS['x_14']) > x_bound)|(abs(X_LS['y_14']) > y_bound)|(abs(X_LS['x_15']) > x_bound)|(abs(X_LS['y_15']) > y_bound)|(abs(X_LS['x_16']) > x_bound)|(abs(X_LS['y_16']) > y_bound)|(abs(X_LS['x_17']) > x_bound)|(abs(X_LS['y_17']) > y_bound)|(abs(X_LS['x_18']) > x_bound)|(abs(X_LS['y_18']) > y_bound)|(abs(X_LS['x_19']) > x_bound)|(abs(X_LS['y_19']) > y_bound)|(abs(X_LS['x_20']) > x_bound)|(abs(X_LS['y_20']) > y_bound)|(abs(X_LS['x_21']) > x_bound)|(abs(X_LS['y_21']) > y_bound)|(abs(X_LS['x_22']) > x_bound)|(abs(X_LS['y_22']) > y_bound)].index
    
    # Delete these row indexes from the learning set
    X_LS.drop(indexNames , inplace=True)
    y_LS.drop(indexNames , inplace=True)
    
    # %% ------------------------------ Features computation -------------------------------- #
    
    # Transform data as pair of players
    X_LS_pairs, y_LS_pairs = make_pair_of_players(X_LS, y_LS)
    
    # Features
    X_LS_pairs["distance"] = compute_distance_(X_LS_pairs)
    m_s,m_o,gk_n,gk_p = teams_parameters(X_LS_pairs)
    X_LS_pairs["mean_team_s"] = m_s
    X_LS_pairs["mean_team_o"] = m_o
    X_LS_pairs["gk_n"] = gk_n
    X_LS_pairs["gk_p"] = gk_p
    X_LS_pairs["side_sender"] = side_sender(X_LS_pairs)
    X_LS_pairs["forward_distance"] = forward_distance(X_LS_pairs)
    X_LS_pairs["lateral_distance"] = lateral_distance(X_LS_pairs)
    X_LS_pairs["intercept_distance"] = intercept_distance(X_LS_pairs)
    zx_sender,zy_sender,zx_j,zy_j = zones(X_LS_pairs)
    X_LS_pairs["zx_sender"] = zx_sender
    X_LS_pairs["zx_j"] = zx_j
    X_LS_pairs["zy_sender"] = zy_sender
    X_LS_pairs["zy_j"] = zy_j
    X_LS_pairs["opponent_distance_sender"] = opponent_distance_sender(X_LS_pairs)
    X_LS_pairs["opponent_distance_receiver"] = opponent_distance_receiver(X_LS_pairs)

    X_features = X_LS_pairs[["distance","mean_team_s","mean_team_o","side_sender","forward_distance","lateral_distance","intercept_distance","opponent_distance_sender","opponent_distance_receiver","zx_sender","zy_sender","zx_j","zy_j","same_team"]]

    # Same transformation as LS
    X_TS_pairs, _ = make_pair_of_players(X_TS)
    
    # Same features computation
    X_TS_pairs["distance"] = compute_distance_(X_TS_pairs)
    mT_s,mT_o,gkT_n,gkT_p = teams_parameters(X_TS_pairs)
    X_TS_pairs["mean_team_s"] = mT_s
    X_TS_pairs["mean_team_o"] = mT_o
    X_TS_pairs["gk_n"] = gkT_n
    X_TS_pairs["gk_p"] = gkT_p
    X_TS_pairs["side_sender"] = side_sender(X_TS_pairs)
    X_TS_pairs["forward_distance"] = forward_distance(X_TS_pairs)
    X_TS_pairs["lateral_distance"] = lateral_distance(X_TS_pairs)
    X_TS_pairs["intercept_distance"] = intercept_distance(X_TS_pairs)
    zxT_sender,zyT_sender,zxT_j,zyT_j = zones(X_TS_pairs)
    X_TS_pairs["zx_sender"] = zxT_sender
    X_TS_pairs["zx_j"] = zxT_j
    X_TS_pairs["zy_sender"] = zyT_sender
    X_TS_pairs["zy_j"] = zyT_j
    X_TS_pairs["opponent_distance_sender"] = opponent_distance_sender(X_TS_pairs)
    X_TS_pairs["opponent_distance_receiver"] = opponent_distance_receiver(X_TS_pairs)

    X_TS_features = X_TS_pairs[["distance","mean_team_s","mean_team_o","side_sender","forward_distance","lateral_distance","intercept_distance","opponent_distance_sender","opponent_distance_receiver","zx_sender","zy_sender","zx_j","zy_j","same_team"]]
    
    # %% ----------------------------- Training on default models------------------------------- #

    with measure_time("SGDClassifier"):          
        model = SGDClassifier()
        model.fit(X_features, y_LS_pairs) 
        scores = cross_val_score(model, X_features, y_LS_pairs, cv=4, scoring= 'roc_auc')
        print("AUC: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    with measure_time("DecisionTreeClassifier"): 
        model = DecisionTreeClassifier()
        model.fit(X_features, y_LS_pairs) 
        scores = cross_val_score(model, X_features, y_LS_pairs, cv=4, scoring= 'roc_auc')
        print("AUC: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    with measure_time("KNeighborsClassifier"):          
        model = KNeighborsClassifier()
        model.fit(X_features, y_LS_pairs) 
        scores = cross_val_score(model, X_features, y_LS_pairs, cv=4, scoring= 'roc_auc')
        print("AUC: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    with measure_time("SVC"): 
        model = SVC(class_weight = 'balanced')
        model.fit(X_features, y_LS_pairs) 
        scores = cross_val_score(model, X_features, y_LS_pairs, cv=4, scoring= 'roc_auc')
        print("AUC: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
       
    with measure_time("LinearSVC"):          
        model = LinearSVC()
        model.fit(X_features, y_LS_pairs) 
        scores = cross_val_score(model, X_features, y_LS_pairs, cv=4, scoring= 'roc_auc')
        print("AUC: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
  
    with measure_time("GaussianNB"):          
        model = GaussianNB()
        model.fit(X_features, y_LS_pairs) 
        scores = cross_val_score(model, X_features, y_LS_pairs, cv=4, scoring= 'roc_auc')
        print("AUC: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    with measure_time("MLPClassifier"): 
        model = MLPClassifier()
        model.fit(X_features, y_LS_pairs) 
        scores = cross_val_score(model, X_features, y_LS_pairs, cv=4, scoring= 'roc_auc')
        print("AUC: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    with measure_time("RandomForestClassifier"): 
        model = RandomForestClassifier(n_estimators = 100,class_weight = 'balanced', criterion = 'entropy', warm_start = False, bootstrap = True)
        model.fit(X_features, y_LS_pairs) 
        scores = cross_val_score(model, X_features, y_LS_pairs, cv=4, scoring= 'roc_auc')
        print("AUC: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
        
    with measure_time("AdaBoostClassifier"):          
        model = AdaBoostClassifier()
        model.fit(X_features, y_LS_pairs) 
        scores = cross_val_score(model, X_features, y_LS_pairs, cv=4, scoring= 'roc_auc')
        print("AUC: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    with measure_time("GradientBoostingClassifier"): 
        model = GradientBoostingClassifier()
        model.fit(X_features, y_LS_pairs) 
        scores = cross_val_score(model, X_features, y_LS_pairs, cv=4, scoring= 'roc_auc')
        print("AUC: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    # %% ------------------- Optimisation of the KNeighbors classifier------------------- #
    n_neighbors =  range(1, 101, 5)
    mean_score = np.zeros( len(n_neighbors) )
    std_score = np.zeros( len(n_neighbors) )

    for i, n in enumerate(n_neighbors):
        model = KNeighborsClassifier(n_neighbors=n)
        cross_score = cross_val_score(model, X_features, y_LS_pairs, cv=5, scoring= 'roc_auc')
        mean_score[i] = np.mean(cross_score)
        std_score[i] = np.std(cross_score)

    n_neighbors_optimal = n_neighbors[np.argmax(mean_score)]
    print("Optimal number of neighbors:")
    print(n_neighbors_optimal)
    plt.figure(dpi=200)
    plt.title("KNN: mean AUC wrt number of neighbors")
    plt.plot(n_neighbors, mean_score)
    plt.xlabel('number of neighbors')
    plt.ylabel('mean AUC')
    plt.savefig('knnAUC.png')
    print("AUC: %0.3f (+/- %0.3f)" % (mean_score[np.argmax(mean_score)], std_score[np.argmax(mean_score)]))
    #model = KNeighborsClassifier(n_neighbors=n_neighbors_optimal)


    # %% ------------------- Optimisation of the Adaboost classifier------------------- #
    n_estimators = range(100, 900, 200)
    mean_score = np.zeros( len(n_estimators) )
    std_score = np.zeros( len(n_estimators) )

    for i, n in enumerate(n_estimators):
        model = AdaBoostClassifier(n_estimators=n)
        cross_score = cross_val_score(model, X_features, y_LS_pairs, cv=4, scoring= 'roc_auc')
        mean_score[i] = np.mean(cross_score)
        std_score[i] = np.std(cross_score)

    n_optimal = n_estimators[np.argmax(mean_score)]
    print(n_optimal)
    plt.figure(dpi=200)
    plt.title("AdaBoost : mean AUC wrt number of estimators")
    plt.plot(n_estimators, mean_score)
    plt.xlabel('number of estimators')
    plt.ylabel('mean AUC')
    plt.savefig('adaboostAUC.png')
    print("AUC: %0.3f (+/- %0.3f)" % (mean_score[np.argmax(mean_score)], std_score[np.argmax(mean_score)]))
    #model = AdaBoostClassifier(n_estimators=n_optimal)
        
    # %% ------------------- Optimisation of the Gradient boosting classifier------------------- #
    gbm0 = GradientBoostingClassifier(random_state=10)
    alg = gbm0
    alg.fit(X_features, y_LS_pairs) 
    features = range(1,15,1)
    feat_imp = pd.Series(alg.feature_importances_, features).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    
    param_test1 = {'n_estimators':range(60,121,10)}
    gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.2, min_samples_split=2000,min_samples_leaf=200,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10), 
    param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    gsearch1.fit(X_features, y_LS_pairs) 
    gsearch1.best_params_, gsearch1.best_score_
    print(gsearch1.best_params_)
    gsearch1.cv_results_
    
    param_test2 = {'max_depth':range(6,9,1), 'min_samples_split':range(1000,1601,200)}
    gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.2, n_estimators=90, max_features='sqrt', subsample=0.8, random_state=10), 
    param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    gsearch2.fit(X_features, y_LS_pairs)
    print(gsearch2.best_params_)
    gsearch2.cv_results_
    
    param_test3 = {'min_samples_split':range(1200,1401,100), 'min_samples_leaf':range(30,71,10)}
    gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.2, n_estimators=90,max_depth=7,max_features='sqrt', subsample=0.8, random_state=10), 
    param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    gsearch3.fit(X_features, y_LS_pairs)
    print(gsearch3.best_params_)
    gsearch3.cv_results_

    alg = gsearch3.best_estimator_
    alg.fit(X_features, y_LS_pairs)
    feat_imp = pd.Series(alg.feature_importances_, features).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    
    param_test4 = {'max_features':range(1,7,1)}
    gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.2, n_estimators=90,max_depth=7, min_samples_split=1400, min_samples_leaf=60, subsample=0.8, random_state=10),
    param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    gsearch4.fit(X_features, y_LS_pairs)
    print(gsearch4.best_params_)
    gsearch4.cv_results_
    
    param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
    gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.2, n_estimators=90,max_depth=7,min_samples_split=1400, min_samples_leaf=60, random_state=10,max_features=3),
    param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    gsearch5.fit(X_features, y_LS_pairs)
    print(gsearch5.best_params_)
    gsearch5.cv_results_
    
    # %% ------------------- Optimisation of the Random forest classifier---------------------- #
    rf0 = RandomForestClassifier(random_state=10)
    rf0.fit(X_features, y_LS_pairs) 
    features = range(1,15,1)
    feat_imp = pd.Series(rf0.feature_importances_, features).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    
    par_test1 = {'max_depth':range(7,14,1)} 
    rfsearch1 = GridSearchCV(estimator = RandomForestClassifier(random_state=10), 
    param_grid = par_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    rfsearch1.fit(X_features, y_LS_pairs)
    print(rfsearch1.best_params_)
    rfsearch1.cv_results_
    
    par_test2 = {'min_samples_split':range(100,401,100)} 
    rfsearch2 = GridSearchCV(estimator = RandomForestClassifier(random_state=10), 
    param_grid = par_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    rfsearch2.fit(X_features, y_LS_pairs)
    print(rfsearch2.best_params_)
    rfsearch2.cv_results_
    
    par_test3 = {'max_leaf_nodes':range(200,401,50)} 
    rfsearch3 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split = 200,max_depth = 13,random_state=10), 
    param_grid = par_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    rfsearch3.fit(X_features, y_LS_pairs)
    print(rfsearch3.best_params_)
    rfsearch3.cv_results_
    
    par_test4 = {'min_samples_leaf':range(10,51,10)} 
    rfsearch4 = GridSearchCV(estimator = RandomForestClassifier(max_leaf_nodes = 350,min_samples_split = 200,max_depth = 13,random_state=10), 
    param_grid = par_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    rfsearch4.fit(X_features, y_LS_pairs)
    print(rfsearch4.best_params_)
    rfsearch4.cv_results_
    
    par_test5 = {'n_estimators':range(50,151,50)} 
    rfsearch5 = GridSearchCV(estimator = RandomForestClassifier(min_samples_leaf = 50,max_leaf_nodes = 350,min_samples_split = 200,max_depth = 13,random_state=10), 
    param_grid = par_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    rfsearch5.fit(X_features, y_LS_pairs)
    print(rfsearch5.best_params_)
    rfsearch5.cv_results_
    
    par_test6 = {'max_features':range(6,10,1)} 
    rfsearch6 = GridSearchCV(estimator = RandomForestClassifier(n_estimators = 100,min_samples_leaf = 50,max_leaf_nodes = 350,min_samples_split = 200,max_depth = 13,random_state=10), 
    param_grid = par_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    rfsearch6.fit(X_features, y_LS_pairs)
    print(rfsearch6.best_params_)
    rfsearch6.cv_results_
    
    par_test7 = {'max_samples':[0.2,0.4,0.6,0.8,0.9]} 
    rfsearch7 = GridSearchCV(estimator = RandomForestClassifier(max_features = 7,n_estimators = 100,min_samples_leaf = 50,max_leaf_nodes = 350,min_samples_split = 200,max_depth = 13,random_state=10), 
    param_grid = par_test7, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    rfsearch7.fit(X_features, y_LS_pairs)
    print(rfsearch7.best_params_)
    rfsearch7.cv_results_
    
    mod = RandomForestClassifier(n_estimators=100,max_depth=7,min_samples_split=1400, min_samples_leaf=60, random_state=10,max_features=3)
    mod.fit(X_features, y_LS_pairs) 
    feat_imp = pd.Series(mod.feature_importances_, features).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    # %% ------------------------- Optimisation of the MLP classifier-------------------------- #
    
    mlp0 = MLPClassifier(max_iter=100,hidden_layer_sizes = (150,100,50))
    
    parameter0 = {'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)]}
    parameters00 = { 'hidden_layer_sizes': [(200,150,100), (250,100,100)]}
    parameters1 = {'activation': ['identity', 'logistic', 'tanh', 'relu']}
    parameters2 = {'solver': ['sgd', 'adam']}
    parameters3 = {'alpha': [0.0001, 0.05]}
    parameters4 = {'learning_rate': ['constant','adaptive']}
    parameters5 = {'max_iter':range(50,201,50)}
    
    clf = GridSearchCV(mlp0, parameters5,scoring='roc_auc', n_jobs=-1, cv=3)
    clf.fit(X_features, y_LS_pairs) 
    print(clf.best_params_)
    print(clf.best_score_)
    clf.cv_results_
    
    # %% ------------------------------ Prediction ------------------------------ #
    
    with measure_time("MLPlassifier"):
        
        #model = GradientBoostingClassifier(learning_rate=0.025, n_estimators=720,max_depth=7,min_samples_split=1400, min_samples_leaf=60, subsample=0.75, random_state=10,max_features=3)
        #♀model = RandomForestClassifier(n_estimators=100,max_depth=7,min_samples_split=1400, min_samples_leaf=60, random_state=10,max_features=3)
        model = MLPClassifier(max_iter=100,hidden_layer_sizes = (150,100,50))
        model.fit(X_features, y_LS_pairs) 
        scores = cross_val_score(model, X_features, y_LS_pairs, cv=5, scoring= 'roc_auc')
        print("AUC: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print(scores.mean())
        
    # Predict
    y_pred = model.predict_proba(X_TS_features)[:,1]
 
    # Deriving probas
    probas = y_pred.reshape(X_TS.shape[0], 22)

    # Estimated score of the model
    n = probas.shape[0]
    max_i = np.zeros(probas.shape[0],)
    max2_i = np.zeros(probas.shape[0],)
    certainty_margin = np.zeros(probas.shape[0],)
    new_proba = np.zeros(probas.shape[0],)
    for i in range(n):
        max_i[i] = max(probas[i])
        max2_i[i] = second_max(probas[i])
        certainty_margin[i] = (max_i[i] - max2_i[i])/4
        new_proba[i] = max_i[i] +  certainty_margin[i]
    predicted_score = new_proba.mean()

    # Making the submission file
    fname = write_submission(probas=probas, estimated_score=predicted_score, file_name="features_model_MLPClassifier")
    print('Submission file "{}" successfully written'.format(fname))
    
