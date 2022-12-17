import ffmpeg

# #takes a directory of images and makes a video out of them
# (
#     ffmpeg
#     .input('run_results/%05d.jpg',  framerate=30)
#     .output('movie.mp4')
#     .run()
# )

video=cv2.VideoWriter('video.mp4',-1,1,(width,height))
for j in range(0,5):
    video.write(img)
video.release()