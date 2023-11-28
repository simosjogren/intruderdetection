import argparse
from src.playVideo import play_video

'''
Main launcher for the process.
Usage: python main.py
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="./data/intrusion.avi", help="path to video")
    args = parser.parse_args()
    play_video(args.video_path)