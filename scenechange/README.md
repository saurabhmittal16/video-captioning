# Frame Selection

A simple script that selects a single frame from each scene of the video. The scene detection is performed using [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) and the frame is selected on the basis of clarity (non-blurriness) of the frame.

## Steps

1. Detect scenes using PySceneDetect and save start and end frame of each scene

2. Calculate non-blurriness score of each frame using `cv2.Laplacian`

3. Find the highest scored frame in each range and save to memory

## Resources

1. [https://pyscenedetect.readthedocs.io/en/latest/](https://pyscenedetect.readthedocs.io/en/latest/)

2. [https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/](https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/)
