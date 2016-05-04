import subprocess as sp
import numpy
from PIL import Image
import motion_detector

FFMPEG_BIN = "ffmpeg"

last_image = None

def process(input_file_name, resolution, output_file_name):

    total_attractiveness = 0
    total_frames = 0

    command = [ FFMPEG_BIN,
                '-i', input_file_name,
                '-f', 'image2pipe',
                '-pix_fmt', 'rgb24',
                '-vcodec', 'rawvideo', '-']
    input_pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=resolution[0]*resolution[1]*3)

    command = [ FFMPEG_BIN,
        '-y', # (optional) overwrite output file if it exists
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', ("%dx%d" % (resolution[0], resolution[1])), # size of one frame
        '-pix_fmt', 'rgb24',
        '-r', '25', # frames per second
        '-i', '-', # The imput comes from a pipe
        '-an', # Tells FFMPEG not to expect any audio
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_file_name ]
    output_pipe = sp.Popen( command, stdin=sp.PIPE, stderr=sp.PIPE)

    while True:
        raw_image = input_pipe.stdout.read(resolution[0]*resolution[1]*3)
        if len(raw_image) > 0:
            image_array = numpy.fromstring(raw_image, dtype='uint8')
            image_array = image_array.reshape((resolution[1],resolution[0],3))
            image = Image.fromarray(image_array, 'RGB')

            attractiveness = get_attractiveness(image)

            if attractiveness >= 0.5:
                output_pipe.stdin.write( raw_image )
                total_frames += 1
                total_attractiveness += attractiveness

            input_pipe.stdout.flush()
        else:
            break

    input_pipe.terminate()
    input_pipe.stdout.close()
    output_pipe.stdin.close()
    output_pipe.stderr.close()
    output_pipe.wait()

    mean_attractiveness = total_attractiveness / total_frames if total_frames > 0 else 0
    total_duration = total_frames / 25

    return {
        'duration': total_duration,
        'attractiveness': mean_attractiveness
    }



def get_attractiveness(image):
    global last_image
    if last_image is None:
        last_image = image
        return 1
    else:
        motion = motion_detector.get_amount_of_motion(image, last_image, 64, 0.0005)
        if motion >= 1:
            last_image = image
            return motion / 2
        else:
            return 0
