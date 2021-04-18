import cv2
from scenedetect import VideoManager
from scenedetect import SceneManager

# For content-aware scene detection:
from scenedetect.detectors import ContentDetector


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


def find_scenes(video_path, threshold=30.0):
    # Create our video & scene managers, then add the detector.
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    # Improve processing speed by downscaling before processing.
    video_manager.set_downscale_factor()

    # Start the video manager and perform the scene detection.
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    # Each returned scene is a tuple of the (start, end) timecode.
    return scene_manager.get_scene_list()


def save_frames(INPUT_PATH, OUTPUT_PATH, name):
    scenes = find_scenes(INPUT_PATH)
    scene_ranges = []
    blurr_score = []

    # Find start and end frame of each scene
    for (start, end) in scenes:
        scene_ranges.append((start.get_frames(), end.get_frames()))

    vidcap = cv2.VideoCapture(INPUT_PATH)
    success, image = vidcap.read()
    count = 0

    # Measure non-blurry score of each frame
    while success:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)

        blurr_score.append(fm)
        success, image = vidcap.read()
        count += 1

    best_frames = []

    # Find frame with max score in each scene range
    for i, (start, end) in enumerate(scene_ranges):
        scene = blurr_score[start:end]
        best_score = max(scene)
        best_frame = scene.index(best_score)
        best_frames.append(start + best_frame)

    vidcap = cv2.VideoCapture(INPUT_PATH)
    success, image = vidcap.read()
    count = 0
    ind = 0

    root = name

    # Save best scored frames for each scene
    while success and ind < len(best_frames):
        if count == best_frames[ind]:
            filename = "{}/{}{}.jpg".format(OUTPUT_PATH, root, ind)
            cv2.imwrite(filename, image)
            ind += 1

        blurr_score.append(fm)
        success, image = vidcap.read()
        count += 1

    return root, ind


if __name__ == "__main__":
    save_frames("./videos/tourism.mp4")
