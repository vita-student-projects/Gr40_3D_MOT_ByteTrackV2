import math
import tools.math_algo as ma
import tools.ByteTrack as ByteTrack
import imageio
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_ego_pose(token, nusc):
    """ search for ego pose with given token and return it"""
    for ego_pose in nusc.ego_pose:
        if ego_pose["token"] == token:
            return ego_pose
    return None 

def rad2deg(angle):
    return (angle * 180 / math.pi)

def find_max_min(list_sample_tokens, gt_boxes, pred_boxes, confidence_threshold=0.4):
    """
    Find max and min x and y values for a given list of sample tokens
    This is usefuls to set the limits of the plot
    gt_boxes and pred_boxes are dictionaries with sample tokens as keys
    """
    max_x, min_x, max_y, min_y = -10000, 10000, -10000, 10000
    for token in list_sample_tokens:
        # get all boxes from sample
        gt_sample_boxes = gt_boxes[token]
        pred_sample_boxes = pred_boxes[token]
        
        for elem in gt_sample_boxes:
            translation = elem["translation"][:-1]
            if translation[0] > max_x:
                max_x = translation[0]
            if translation[0] < min_x:
                min_x = translation[0]
            if translation[1] > max_y:
                max_y = translation[1]
            if translation[1] < min_y:
                min_y = translation[1]


        for elem in pred_sample_boxes:
            if elem["detection"] > confidence_threshold:
                translation = elem["translation"][:-1]
                if translation[0] > max_x:
                    max_x = translation[0]
                if translation[0] < min_x:
                    min_x = translation[0]
                if translation[1] > max_y:
                    max_y = translation[1]
                if translation[1] < min_y:
                    min_y = translation[1]

    return max_x, min_x, max_y, min_y

def display_detections_and_tracks_bev_V3(nusc, gt_list, pred_list, token, max_x, min_x, max_y, min_y, sample_tracks, confidence_threshold=0.4, verbose=False, index = 69, plot_gt = True, plot_pred = True, plot_track = True):
    """ display BEV detections for a given sample token (gt in red, pred in blue, track in magenta) """
    fig, ax = plt.subplots(figsize=(15,15))
    ax.autoscale()
    ax.set_aspect('equal')
    # plt.suptitle(f"BEV detections for sample {token}")
    plt.suptitle(f"frame {index}")
    plt.title(f"confidence thresh = {confidence_threshold}, red = gt, blue = pred, magenta = track")
        # plot ground truth boxes
    if plot_gt:
        for elem in gt_list:
            if elem["detection_name"] not in ByteTrack.NUSCENES_TRACKING_NAMES:
                continue
            translation = elem["translation"][:-1]
            size = elem["size"][:-1]
            rotation = elem["rotation"]
            roll, pitch, yaw = ma.euler_from_quaternion(rotation[1], rotation[2], rotation[3], rotation[0])
            if verbose:
                print(f"gt roll = {roll}, pitch = {pitch}, yaw = {yaw}")
            rect = patches.Rectangle((translation[0], translation[1]), size[0], size[1], angle=yaw, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    if plot_pred:
        # plot predictions boxes
        for elem in pred_list:
            if elem["detection_name"] not in ByteTrack.NUSCENES_TRACKING_NAMES:
                continue
            if elem["detection_score"] > confidence_threshold:
                translation = elem["translation"][:-1]
                size = elem["size"][:-1]
                rotation = elem["rotation"]
                roll, pitch, yaw = ma.euler_from_quaternion(rotation[1], rotation[2], rotation[3], rotation[0])
                if verbose:
                    print(f"pred roll = {roll}, pitch = {pitch}, yaw = {yaw}")
                rect = patches.Rectangle((translation[0], translation[1]), size[0], size[1], angle=yaw, linewidth=1, edgecolor='b', facecolor='none')
                ax.add_patch(rect)
    # plot ego vehicle
    ego_token = nusc.get("sample_data", nusc.get('sample', token)["data"]["CAM_FRONT"])["ego_pose_token"] # we use CAM_FRONT but it could be any camera/lidar
    ego_pose_dict = get_ego_pose(ego_token, nusc)
    translation = ego_pose_dict["translation"][:-1]
    rotation = ego_pose_dict["rotation"]
    roll, pitch, yaw = ma.euler_from_quaternion(rotation[1], rotation[2], rotation[3], rotation[0])
    if verbose:
        print(f"ego vehicle roll = {roll}, pitch = {pitch}, yaw = {yaw}")
    rect = patches.Rectangle((translation[0], translation[1]), 1.7, 4, angle=yaw, linewidth=1, edgecolor='g', facecolor='none') # dimensions of renault zoe :)))))))))))))
    ax.add_patch(rect)
    # plot tracks
    if plot_track:
        for track in sample_tracks:
            # print(track)
            if track["Found"] == False:
                color = "red"
            else:
                color = "black"
            x, y = track['state'][0], track['state'][1]
            w, h = track["state"][3], track["state"][4]
            yaw = track["state"][6]
            # print(yaw)
            # track number
            plt.text(x, y, track['track_id'], fontsize=12, color=color)
            # track box
            rect = patches.Rectangle((x, y), w, h, angle=yaw, linewidth=1, edgecolor='m', facecolor='none')
            ax.add_patch(rect)
    # plot limits
    border = 10
    ax.set_xlim(min_x-border, max_x+border)
    ax.set_ylim(min_y-border, max_y+border)

def find_max_min(list_sample_tokens, gt_boxes, pred_boxes, confidence_threshold=0.4):
    """
    Find max and min x and y values for a given list of sample tokens
    This is usefuls to set the limits of the plot
    gt_boxes and pred_boxes are dictionaries with sample tokens as keys
    """
    max_x, min_x, max_y, min_y = -10000, 10000, -10000, 10000
    for token in list_sample_tokens:
        # get all boxes from sample
        gt_sample_boxes = gt_boxes[token]
        pred_sample_boxes = pred_boxes[token]
        
        for elem in gt_sample_boxes:
            translation = elem["translation"][:-1]
            if translation[0] > max_x:
                max_x = translation[0]
            if translation[0] < min_x:
                min_x = translation[0]
            if translation[1] > max_y:
                max_y = translation[1]
            if translation[1] < min_y:
                min_y = translation[1]


        for elem in pred_sample_boxes:
            if elem["detection_score"] > confidence_threshold:
                translation = elem["translation"][:-1]
                if translation[0] > max_x:
                    max_x = translation[0]
                if translation[0] < min_x:
                    min_x = translation[0]
                if translation[1] > max_y:
                    max_y = translation[1]
                if translation[1] < min_y:
                    min_y = translation[1]

    return max_x, min_x, max_y, min_y

def create_gif(ordered_sample_tokens, nusc, gt, pred, history, confidence_threshold, verbose = False, plot_gt=True, plot_pred=True, plot_track=True):
    """ Create a GIF file with the plots of the BEV detections for a given list of sample tokens"""
    tokens = ordered_sample_tokens

    # Create a list to store the plots
    plots = []

    # Find the max and min x and y values
    max_x, min_x, max_y, min_y = find_max_min(tokens, gt, pred, confidence_threshold=confidence_threshold)

    # Create and save each plot
    for i, token in enumerate(tokens):
        gt_list = gt[token]
        pred_list = pred[token]
        display_detections_and_tracks_bev_V3(nusc, 
                                             gt_list, 
                                             pred_list, 
                                             token, 
                                             confidence_threshold=confidence_threshold, 
                                             verbose=verbose, 
                                             max_x=max_x, 
                                             min_x=min_x, 
                                             max_y=max_y, 
                                             min_y=min_y, 
                                             index = i, 
                                             sample_tracks=history[i], 
                                             plot_gt=plot_gt, 
                                             plot_pred=plot_pred, 
                                             plot_track=plot_track)

        # Save the plot to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Read the image from the BytesIO object and append it to the list of plots
        image = imageio.imread(buf)
        plots.append(image)
        
        # Close the figure to free up memory
        plt.close()

    # Save the list of plots as a GIF file
    imageio.mimsave('plots.gif', plots, duration=0.5)