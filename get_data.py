import cv2
import os
import subprocess

def video_to_frames(video_path, video_name):
    video = cv2.VideoCapture(video_path)
    success, image = video.read()
    count = 0

    # Save Frames into Train and Test Folders
    # 7 Seconds of frames at 30fps -> Train
    # 3 Seconds of frames at 30fps -> Test
    # Get frames from 0:20-0:30
    while success and count <30*30:
        if count > 20*30 and count < 27*30: cv2.imwrite(os.path.join(video_name,'train/%d.png' % count), image)
        elif count >= 27*30: cv2.imwrite(os.path.join(video_name,'test/%d.png' % count), image)
        success, image = video.read()

        count+=1

def foveate_frames():
    frame_dir = ['test_vid/train/', 'test_vid/test/']
    run_foveate = 1

    for dir in frame_dir:
        frames = glob.glob(os.path.join(dir, '*.png'))
        for frame in frames:
            name = frame[len(dir):-4]
            foveate = 'python /content/Image_Foveation_Python/retina_transform.py /content/{}/{}.png'.format(dir, name).split(' ')
            if run_foveate==1: subprocess.call(foveate)

        frames = [int(frame[len(dir):-4]) for frame in frames]
        print('Starting Frame:', min(frames), 'Ending Frames:', max(frames))

def extract_frames(video_file='test_vid.mp4'):
    video_name = video_file[:-4]
    if not os.path.isdir(video_name): os.mkdir(video_name)
    if not os.path.isdir(os.path.join(video_name,'test')):
      os.mkdir(os.path.join(video_name,'test'))
      os.mkdir(os.path.join(video_name,'train'))
    video_to_frames(video_file, video_name)

    print( len(glob.glob(os.path.join(os.path.join(video_name,'test'), '*.png'))) )

def main():
    extract_frames()
    foveate_frames()

if __name__ == '__main__':
    main()
