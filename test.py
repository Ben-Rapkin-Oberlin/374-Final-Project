import ffmpeg
(
    ffmpeg
    .input('images/protest/%05d.jpg',  framerate=30)
    .output('movie.mp4')
    .run()
)
#.input('images/protest/*.jpg', pattern_type='glob', framerate=30)
#-i image%03d.jpg