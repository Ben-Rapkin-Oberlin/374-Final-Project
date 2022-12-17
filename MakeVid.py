import ffmpeg

#takes a directory of images and makes a video out of them
(
    ffmpeg
    .input('run_results/%05d.jpg',  framerate=30)
    .output('movie.mp4')
    .run()
)