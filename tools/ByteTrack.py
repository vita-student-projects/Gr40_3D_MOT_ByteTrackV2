#----Imports for manipulating the dataset----#
import pandas as pd 
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common.loaders import load_prediction, load_gt
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.render import visualize_sample, dist_pr_curve
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.tracking.data_classes import TrackingBox

#----Other imports----#
import numpy as np
from numpy import *
import json
from pyquaternion import Quaternion
import copy

#----Imports for visualization tool----#
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#----Local imports----#
import tools.math_algo as ma
from tools.Kalman import KalmanFilter
import tools.covariance as covariance

#----Global variables----#

TRACKING_PATH              = "data/tracking/"
TRAINVAL_PATH              = "data/trainval/"
TEST_PATH                  = "data/test/"

NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck'
  ]

DT = 0.5

#----Functions----#
def get_sample_token_sequence(nusc):
    """ Returns a list of list of sample tokens for each scene """
    sample_token_sequences = []
    for scene in nusc.scene:
        sample_token_sequence = []
        sample_token = scene['first_sample_token']
        while sample_token:
            sample_token_sequence.append(sample_token)
            sample = nusc.get('sample', sample_token)
            sample_token = sample['next']
        sample_token_sequences.append(sample_token_sequence)
    return sample_token_sequences


def association(Detections, Tracklets, hung_thresh):
    """ Takes as input all the current detectiosn and tracklets in a list format
    Return the newly updated tracklets, the unmatched detections and the unmatched tracklets"""
    if len(Detections) == 0 or len(Tracklets) == 0:
        similarity_matrix = [[]]
    else:
        similarity_matrix = []
        for detection in Detections:
            similarity_row = []
            for tracklet in Tracklets:
                if tracklet['Found'] == False:
                    back_pred = backward_prediction(detection) # compute the backward prediction of the detection
                    similarity_row.append(1-(1+ma.GIoU(back_pred, tracklet))/2) # compute similarity between all tracklets and a given backward prediciton
                else:
                    similarity_row.append(np.inf) # if we dont do this the indices wont match when we check the pairs
            similarity_matrix.append(similarity_row) # repeat for all detections
    matches, unmatched_det_index, _ = ma.hungarian_algorithm(np.array(similarity_matrix), thresh=hung_thresh) # solve the linear assignment problem
    unmatched_detections = []
    for matched_pair in matches: # for each matched pair, update the tracklet with the detection
        det = Detections[matched_pair[0]]
        track = Tracklets[matched_pair[1]]
        velocity = det["velocity"]
        det_state = format_det_state(det) # format the detection
        track['state'] = det_state # update the tracklet with the detection state
        track['Found'] = True # found it
        track['nb_lost_frame'] = 0 # set the number of lost frame to 0 as it has been found
        track['kalman'].update(det_state, det['detection_score'])
        track['velocity'] = velocity
        track['tracking_score'] = det["detection_score"]
    if len(Detections) == 0:
        unmatched_detections = []
    else:
        for unmatched_det in unmatched_det_index:
            unmatched_detections.append(Detections[unmatched_det])

    return unmatched_detections


def backward_prediction(detection):  
    """returns the x, y position of the object in the previous frame
    based on the velocities detected in the current frame"""
    back_pred = format_det_state(detection)
    back_pred[0] = back_pred[0] - back_pred[-3]*DT
    back_pred[1] = back_pred[1] - back_pred[-2]*DT
    return back_pred 


def format_det_state(det):
    """transforms the detcions into a state vector that can be used by the Kalman filter"""
    _, _, orientation = ma.euler_from_quaternion(det["rotation"][1], det["rotation"][2], det["rotation"][3], det["rotation"][0])
    
    state = [det["translation"][0], det["translation"][1], det["translation"][2], \
            det["size"][0],        det["size"][1],        det["size"][2],      orientation, \
            det["velocity"][0],    det["velocity"][1],    0]
    return state


def commonelems(x,y):
    """returns True if there is at least one common element between x and y"""
    common=False
    for value in x:
        if value in y:
            common=True
    return common

def get_common_tokens(token_list, pred):
    """returns a list of list of sample tokens for each scene that have detections in the pred json"""
    val_token_list = []
    for scene_tokens in token_list:
        if commonelems(scene_tokens, list(pred.keys())):
            val_token_list.append(scene_tokens)
    return val_token_list


def format_to_nuscene(sample_token, tracks):
  """formats the tracks into a dictionary that can be used by the nuscene evaluation tool"""
  rotation = ma.quaternion_from_euler(0,0,tracks['state'][6])
  
  sample_result = {
    'sample_token': sample_token,
    'translation': [tracks['state'][0], tracks['state'][1], tracks['state'][2]],
    'size': [tracks['state'][3], tracks['state'][4], tracks['state'][5]],
    'rotation': [rotation[0], rotation[1], rotation[2], rotation[3]],
    'velocity': [0, 0], 
    'tracking_id': str(int(tracks['track_id'])),
    'tracking_name': tracks['tracking_name'],
    'tracking_score': tracks['tracking_score']
  }

  return sample_result


def load_data(dataset=None, detection_path=None, eval_split=None, verbose=True):
    '''
    Dataset can be either "trainval" or "test", detection_path is the path to the detection folder.
    eval_split is the split of the dataset to evaluate on, can be either "train", "val" or "test".
    '''
    if dataset =="trainval":
        nusc = NuScenes(version='v1.0-trainval', dataroot=TRAINVAL_PATH, verbose=verbose)
    elif dataset =="test":  
        nusc = NuScenes(version='v1.0-test', dataroot=TEST_PATH, verbose=verbose)
    else:
        raise ValueError("Dataset must be either 'trainval' or 'test'")
    token_list = get_sample_token_sequence(nusc)

    if verbose:
        print("⚙️Loading predictions...⚙️")
    res_boxes_full = load_prediction(result_path=detection_path, max_boxes_per_sample=300, box_cls=DetectionBox, verbose=True)
    res_boxes = res_boxes_full[0]
    pred = res_boxes.serialize()
    if verbose:
        print("✅Predictions succesfully loaded✅")

    if eval_split != 'test':
        if verbose:
            print("⚙️Loading ground truth...⚙️")
        gt_boxes = load_gt(nusc, eval_split = eval_split, box_cls=DetectionBox, verbose=True)
        gt = gt_boxes.serialize() 
        if verbose:
            print("✅Ground truth succesfully loaded✅")
    else:
        gt = None
            

    # we want to make sure that we only keep the samples that have detections in the detections val json
    val_token_list = []
    for scene_tokens in token_list:
        if commonelems(scene_tokens, list(pred.keys())):
            val_token_list.append(scene_tokens)
            
    return nusc, gt, pred
     
        
def convert_history_to_dict(scene, history):
    """converts the history into a dictionary that can be used by the nuscene evaluation tool"""
    temp = {}
    for i, token in enumerate(scene):
        tracks = []
        for track in history[i]:
            tracks.append(format_to_nuscene(token, track))
        temp[token] = tracks
    return temp


def save_to_json(results, str):
    """saves the history to a json file"""
    print(f"🔐Saving history to {str}🔐")
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)
        
    big_dic = {}
    big_dic['meta'] = {'use_camera': True, 'use_lidar': False, 'use_radar': False, 'use_map': False, 'use_external': False}
    big_dic['results'] = results

    # save history to json    
    with open(str, 'w') as fp:
        json.dump(big_dic, fp, cls=NumpyEncoder)
    
        
def ByteTrack(sample_tokens, pred, track_index, confidence_threshold, hung_thresh, verbose = False):  
    """ ByteTrack algorithm 
    sample_tokens: list of sample tokens for a given scene
    pred: loaded predictions for a given scene (dictionarry with sample tokens as keys)
    track_index: index of the first tracklet (if we have multiple scenes we want to always have a different id)
    confidence_threshold: confidence threshold for the detections separtation in Dhigh and Dlow
    hung_thresh: threshold for the hungarian algorithm
    verbose: if True, prints the progress of the algorithm over all the frames and prints track deletions"""
    Covariance = covariance.Covariance()
    global tracks # les globals servent à rien dans notre cas non ?
    tracks = [] 
    global history
    history = []

    if verbose:
        print('⌛Start tracking ⌛')
    for i, token in enumerate(sample_tokens): # ≡ for frame in video
        if verbose:
            print("⌛Frame", i, "⌛")
        if i == 0:
            tracks = [] 
        Dhigh = []
        Dlow = []
        detections = pred[token] # Get the boxes for the sample
        for detection in detections:
            if detection["detection_name"] not in NUSCENES_TRACKING_NAMES:
                continue
            if detection["detection_score"] > confidence_threshold:
                Dhigh.append(detection)
            else:
                Dlow.append(detection)


        for track in tracks:
            track['state_predict'] = track['kalman'].predict()
            track['Found'] = False
        
        # First association
        unmatched_detections_1 = association(Dhigh, tracks, hung_thresh=hung_thresh)
        
        # Second association
        _ = association(Dlow, tracks, hung_thresh=hung_thresh) 
        
        # If the tracks hasn't been spotted in over 6 frames, A.K.A. 3 sec with 2 fps, we delete it.
        for track in tracks:
            if track['nb_lost_frame'] > 6: 
                if verbose : print("❌track deleted🗑️")
                tracks.remove(track)
            elif track['Found'] == False:
                # As we didn't find a matching detection, we update the state with the predicted state from KF
                track['state'][:-3] = track['state_predict'] 
                track['nb_lost_frame'] += 1

        # We create new tracks for the remaining detections from Dhigh that haven't been matched.
        for d in unmatched_detections_1:
            track_index += 1
            tracking_name = d['detection_name']
            state = format_det_state(d)
            tracks.append({'state' : state, 
                        'tracking_name' : tracking_name,  
                        'track_id' : track_index,
                        'kalman' : KalmanFilter(state, DT, tracking_name, Covariance),
                        'state_predict': None,
                        'nb_lost_frame' : 0,
                        'Found' : False,
                        'velocity': d["velocity"],
                        'tracking_score': d["detection_score"]}) 
        tracks_copy = copy.deepcopy(tracks) # need to do a deepcopy because otherwise it's a shallow copy and the history will be the same for all the frames
        history.append(tracks_copy)
    if verbose: print("✅Tracking for this sequence done✅")

    for key in history:
        for track in key:
            del track['kalman']
    return history, track_index


#--------------------Execution part--------------------#

def init(dataset, detection_path, eval_split, output_name, confidence_threshold=0.4, hungarian_threshold=0.6):
    print("🔐Loading data🔐")
    nusc, gt, pred = load_data(dataset, detection_path, eval_split)
    val_token_list = []
    results = {} 
    track_index = 0
    token_list = get_sample_token_sequence(nusc)
    for scene_tokens in token_list:
        if commonelems(scene_tokens, list(pred.keys())):
            val_token_list.append(scene_tokens)
    for scene_index in range(len(val_token_list)):
        scene = val_token_list[scene_index]
        
        history, track_index = ByteTrack(scene, pred ,track_index, confidence_threshold, hungarian_threshold, verbose = False)
        results = {**results, **convert_history_to_dict(scene, history)}
    save_to_json(results, output_name+".json")

