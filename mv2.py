import ffmpeg
(
    ffmpeg
    .input('images/protest/%05d.jpg',  framerate=30)
    .output('movie.mp4')
    .run()
)