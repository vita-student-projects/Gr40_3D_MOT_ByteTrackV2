# Our version of ByteTrack-V2
## Abstract

In the context of our course, at the [EPFL](www.epfl.ch), CIVIL-459, Deep learning for autonomous vehicle, we were tasked to do a 3D Mulit-object tracking using Monocular camera. One of the goals of this project was either to improve something existing, or to  make a contribution to the subjet. Another goal was for it to be state of the art (SOTA). 
We present our version of ByteTrack V2, and provide all of the necessary files to use it. We also present the tools we setup in order to understand and evaluate our implementation. 

## Our approach

Dealing with trackers is a challenging task as this is a 2-step algorithm. The first part consist of detecting the objects in the scene. Those detections can be either 2D or 3D and are often represented as bounding boxes around the interest identitites. Then comes the second part which consists of tracking the detections by assigning the relevant ones with a unique ID and tracking them across multiple frames. We decided to concentrate only on the second part and use pre-computed detections as input of our implementation. 

After a bit of research about various tracker, we have decided to work on [ByteTrack](https://github.com/ifzhang/ByteTrack)\[2\]. At the time of our project, ByteTrack V2 was visible on the nuscene leaderboard, but the github wasn't available, so our first contribution was to reproduce ByteTrack-V2 and make it available to the public. We followed and analysed their paper, in order to understand how the algorithm worked, and then proceeded to rewrite all the code from scratch.

ByteTrack v2 is the evloution of the popular ByteTrack and is adapted for 3D multi object tracking. It still is based on the idea that every detection boxes should be associated. The boxes are nevertheless separated in two distinct groupes Dhigh and Dlow which contains respecitvely detections above and under a certain confidence threshold. 
For more details please refer to [BytetrackV2](https://arxiv.org/abs/2303.15334).

The detections on the training and validation set of nuscenes were the ones from MEGVII\[4\] (lidar based), that were provided by nuScene\[1\]. Those detections were the ones we mostly used to conduct our experiments. Thanks to another group we also got access to the detection from BEVFormer that is using camera only. 

We brought some modification to the original algorithm, as it may increase slightly the performance, or the speed of execution.

- We removed the fact that there are multiple array for the tracks, and simply added a flag `Found` to each tracklet. When  `tracks['Found']==True`, the algorithm will immediately set the corresponding similarity value to infinity, instead of computing it.
- At the same time in the code, we check the tracking_name. If `tracks['tracking_name']==detection['detection_name'], we proceed. Otherwise, we don't. This is a shortcut, and a security, if one is labelled as a truck, and the other one as a pedestrian, they **shouldn't** be matched, so we'll avoid computing their similarity.
- When implementing a Kalman's filter, one must set the values of the matrices P, Q and R. This is usually either done by trial and error, set with simple values, or computed. ByteTrack\[2\] propose to update the values of R adaptively, using the detection score. We also found that Probabilistic 3D MOT\[3\] have shared their files, in which they have computed the P, Q and R values, for each class in nuScene. As the dataset has not changed, we have opted for reusing their found values in our code. In addition, we have implemented the R update from ByteTrack  \[2\]. With these values, we had multiple possibilities. ByteTrack's paper doesn't provide any kalman's covariance value. So we could either fully use the values found from \[3\], or add the R update from ByteTrack's paper. In addition, we've tested for 3 different values of  	$\alpha$: 75, 100 and 125. Please note that these values are for camera based method. ByteTrack's paper recommend a value of 10 if you are using a Lidar based method.
- We used the [Jonker-Volgenant algorithm](https://github.com/src-d/lapjv) \[8\] instead of the hungarian algorithm. It remains a 0(n^3) complexity but is faster than the hungarian in practice.


## Dataset

We use the NuScenes dataset for this projet. We do not train on it as ByteTrack is not a deep learning approach. Nevertheless, we need the metadata for the trainval set to be able to evaluate our detector and display ground truths in our visualisation tool. The metadata can be downloaded on [this](https://www.nuscenes.org/nuscenes) page. The resulting folder has to be placed in the data/trainval/ folder. You can also download the test metadata (although it doesn't contain the annotations) on the same page and place it in data/test/.

Scenes of this dataset contain 40 samples. The samples are annotated keyframes recorded at 2hz.

We interact with the dataset by creating nuscenes objects using the nuscenes.nuscenes.NuScenes() constructor. From there you can access any element of the dataset using its unique token. 

Loaders are also available such as nuscenes.eval.common.loaders.load_prediction(). We use this particular function to load the detections (with the correct format - see the format section) and then access each sample with its respective token. The ground truths are loaded using nuscenes.eval.common.loaders.load_gt().

The nuscenes object can also be used to fetch the sample tokens in the correct order for a particular scene as a particular sample usually contains the token of the previous and the next sample in the same scene.

## Vizualisation tool

In order for us to visualize our results, but also the quality of the detections in a quantitative way, we developped a visualisation tool that can create gifs of particular scenes. It is able to display the ground truth detection, the detections of the provided detector and the tracks and their unique ID. All in a bird's eye view setup for easy comprehension.

In red are the ground truth bounding boxes provided directly by the nuscenes dataset.
In blue are the detections of the detector. We are able to change the confidence threshold of the detections and make more or less of them present on the gif.
In magenta are the tracks with their unique ID attached to them. The boxes overlap the detections as they are directly copied in ByteTrackV2's association. The ID becomes red when the track is lost by the algorithm and the Kalman filter kicks in.
Finally in green is the ego vehicule.

We show here examples of the detections from MEGVII we use as inputs. We showcase different thresholds (the detection score above which the detections are drawn on the frame).

Here with a threshold of 0 (all the detections are drawn)
![ ](https://github.com/GKrafft2/DLAV/blob/main/gifs/gif_thresh_00.gif)

Here with a threshold of 0.4, there are way less detections and the ones reminaing are decend with regard to the ground truth.
![ ](https://github.com/GKrafft2/DLAV/blob/main/gifs/gif_thresh_04.gif)

Here with a threshold of 0.9 there is only a few detections remaining (that are not necessarly the best)
![ ](https://github.com/GKrafft2/DLAV/blob/main/gifs/gif_thresh_09.gif)

We see that overall the detections are very noisy and there is a lot of low confidence detections that make no sense.

The following gif contains the tracks for  the same scene as before
![ ](https://github.com/GKrafft2/DLAV/blob/main/gifs/gif_track_index_13.gif)

Here are more examples:
![ ](https://github.com/GKrafft2/DLAV/blob/main/gifs/gif_track_index_51.gif)
![ ](https://github.com/GKrafft2/DLAV/blob/main/gifs/gif_track_index_124.gif)
This last gif highlights the role of the detector in the tracking process. If the detector creates detections with a high enough confidence, even if there is no ground truth, the tracker will create a tracklet and try to track it across mulltiple frames.

## Format

Our code is designed to take as input detections in the nuscenes detection submission format.

```
submission {
	"meta": {
		"use_camera":   <bool>          -- Whether this submission uses camera data as an input.
		"use_lidar":    <bool>          -- Whether this submission uses lidar data as an input.
		"use_radar":    <bool>          -- Whether this submission uses radar data as an input.
		"use_map":      <bool>          -- Whether this submission uses map data as an input.
		"use_external": <bool>          -- Whether this submission uses external data as an input.
	},
	"results": {
		sample_token <str>: List[sample_result] -- Maps each sample_token to a list of sample_results.
	}
}
```

```
sample_result {
	"sample_token":       <str>         -- Foreign key. Identifies the sample/keyframe for which objects are detected.
	"translation":        <float> [3]   -- Estimated bounding box location in m in the global frame: center_x, center_y, center_z.
	"size":               <float> [3]   -- Estimated bounding box size in m: width, length, height.
	"rotation":           <float> [4]   -- Estimated bounding box orientation as quaternion in the global frame: w, x, y, z.
	"velocity":           <float> [2]   -- Estimated bounding box velocity in m/s in the global frame: vx, vy.
	"detection_name":     <str>         -- The predicted class for this sample_result, e.g. car, pedestrian.
	"detection_score":    <float>       -- Object prediction score between 0 and 1 for the class identified by detection_name.
	"attribute_name":     <str>         -- Name of the predicted attribute or empty string for classes without attributes.
					   See table below for valid attributes for each class, e.g. cycle.with_rider.
					   Attributes are ignored for classes without attributes.
					   There are a few cases (0.4%) where attributes are missing also for classes
					   that should have them. We ignore the predicted attributes for these cases.
}
```
We also designed a function to transform the history output of ByteTrackV2 to the nuscenes tracking format in order to evaluate our tracking using nuscene's TrackingEval() tool. The sample_result is slightly different:

```
sample_result {
	"sample_token":   <str>         -- Foreign key. Identifies the sample/keyframe for which objects are detected.
	"translation":    <float> [3]   -- Estimated bounding box location in meters in the global frame: center_x, center_y, center_z.
	"size":           <float> [3]   -- Estimated bounding box size in meters: width, length, height.
	"rotation":       <float> [4]   -- Estimated bounding box orientation as quaternion in the global frame: w, x, y, z.
	"velocity":       <float> [2]   -- Estimated bounding box velocity in m/s in the global frame: vx, vy.
	"tracking_id":    <str>         -- Unique object id that is used to identify an object track across samples.
	"tracking_name":  <str>         -- The predicted class for this sample_result, e.g. car, pedestrian.
                                       Note that the tracking_name cannot change throughout a track.
	"tracking_score": <float>       -- Object prediction score between 0 and 1 for the class identified by tracking_name.
                                       We average over frame level scores to compute the track level score.
                                       The score is used to determine positive and negative tracks via thresholding.
}
```
## Experimental Setup & Results

We tested our implementation using two detectors on the validation set. Those are MEGVII and BEVFormer. 
|  metrics | NDS  |  mAP |
| :------------: | :------------: | :------------: |
| MEGVII  |  62,8 | 51,9  |
|  BEVFORMER |  51,7 | 41,6  |

We tested multiple thresholds for the Jonker-Volgenant algorithm (score in the similarity matrix above this value wont be considered for the optimization) and for the Dhigh and Dlow separation.  From there we used the best results to run vor evaluation and get final results.

Results for the Jonke-Volgenant algorithm threshold
 ![](https://github.com/GKrafft2/DLAV/blob/main/graphs/megvii_hung_0109.png)

Results for the Dhigh/Dlow threshold
 ![](https://github.com/GKrafft2/DLAV/blob/main/graphs/megvii_conf_0109.png)
 
Following these 2 graphs, we've settled on a value of 0.4 for the confidence threshold ( for the Dhigh and Dlow), and a value of 0.6 for the  Jonker-Volgenant algorithm.

We also tested different values of alpha as well as not updating the R  matrix in the Kalman filter.  and we've settled on a  XXXXX

Results for the Kalman filter experiments
![](https://github.com/GKrafft2/DLAV/blob/main/graphs/megvii_kalman_var.png)



## How to run

For this, you will need anaconda installed. You can follow the instructions [here](https://www.anaconda.com/download). Then, in anaconda's terminal, run the following:

	cd PATH/YOU/WANT
	git clone https://github.com/GKrafft2/DLAV.git
	conda create --name DLAV python=3.9.16	
	conda activate DLAV
	cd DLAV
	pip install -r requirements.txt

Install nuScene:

	git clone https://github.com/nutonomy/nuscenes-devkit.git
	cd nuscenes-devkit
 	pip install -r setup/requirements.txt

There are multiple notebook that can help you run our code. Mainly, [ByteTrack.ipynb](https://github.com/GKrafft2/DLAV/blob/main/ByteTrack.ipynb "ByteTrack.ipynb"). Simply running the cell should provide you with a json with the `output_name`. You will also need to provide the files for the detections in the same manner as described bellow

```shell script
└── data
│	├──detection_megvii (or any detections of your likings)
│	│	└── megvii_val.json
│	├── trainval
│	│	├── maps
│	│	├── samples
│	│	└── v1.0-trainval
│   	├── test
│	│	├── maps
│	│	└── v1.0-test
│   	├── tracking
│   	├── evaluation_results
│   	└── config
```


Additionnaly, you can run XXX notebook if you want to be provided with visual output in the form of the gifs presented above.

We do not provide any dataset.py, train.py or inference.py, as our method do not include any learning process.

## Conclusion

We provide you our version of the ByteTrack, fully open-source, as mentionned in the goal of this project. This method is extremely performant, but hasn't been tested on current SOTA detectors, despite our best effort and collaboration with other detector's developper, such as [BEVDET](https://github.com/HuangJunJie2017/BEVDet "BEVDET"), for instance. We leave this up to anyone who would like to evaluate this model, against the very best.

## Acknowledgement

special dedicasse vlad


## References

- \[1\] *"nuScenes: A multimodal dataset for autonomous driving"*, Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom, arXiv:1903.11027,
- \[2\] *"ByteTrack: Multi-Object Tracking by Associating Every Detection Box"*, Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, Xinggang Wang, arXiv 2110.06864
- \[3\] *"Probabilistic 3D Multi-Object Tracking for Autonomous Driving"*, Chiu, Hsu-kuang and Prioletti, Antonio and Li, Jie and Bohg, Jeannette, arXiv:2001.05673
- \[4\] *"Class-balanced Grouping and Sampling for Point Cloud 3D Object Detection"*, Zhu, Benjin and Jiang, Zhengkai and Zhou, Xiangxin and Li, Zeming and Yu, Gang, arXiv:1908.09492"
- \[5\] *"3D-IoU-Python"*, https://github.com/AlienCat-K/3D-IoU-Python/tree/master
- \[6\] *"Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression"*, Hamid Rezatofighi, Nathan Tsoi, JunYoung Gwak, Amir Sadeghian, Ian Reid, Silvio Savarese1, arXiv:1902.09630v2
- \[7\] *"TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers"*, Xuyang Bai and Zeyu Hu and Xinge Zhu and Qingqiu Huang and Yilun Chen and Hongbo Fu and Chiew-Lan Tai, arXiv:2203.11496
- \[8\] *" Linear Assignment Problem solver using Jonker-Volgenant algorithm "*,  https://github.com/src-d/lapjv
