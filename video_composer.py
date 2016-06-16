import video_processor
import subprocess

def compose(file_count, output_file_path, config):
    meta = []

    maximum_duration = config["maximum_video_length"]

    # process and rate each video

    for i in range(1, file_count):
        video_meta = video_processor.shorten(
            "tmp/video%s.h264" % i,
            (1280, 720),
            "tmp/processed/video%s.h264" % i,
            config["shortening"]
        )
        video_meta['number'] = i
        if video_meta['attractiveness'] > 0:
            meta.append(video_meta)

    if len(meta) == 0:
        return False

    # pick most interesting

    meta = sorted(meta, key=lambda k: k['duration'] + k['attractiveness'] * 10)
    selected_video_numbers = []
    duration = 0
    attractiveness = 0
    while True and len(meta) > 0:
        video_meta = meta.pop()
        if duration + video_meta['duration'] < maximum_duration:
            selected_video_numbers.append(video_meta['number'])
            duration += video_meta['duration']
            attractiveness += video_meta['attractiveness'] * video_meta['duration']
        else:
            break

    if duration == 0:
        return False

    # postprocess selected

    if "perspective_correction" in config:
        for i in selected_video_numbers:
            video_processor.postprocess(
                "tmp/processed/video%s.h264" % i,
                (1280, 720),
                "tmp/processed/video%s.h264" % i,
                config["perspective_correction"]
            )

    # join into one

    rating = round(attractiveness * 100 / duration)

    video_list = open("tmp/list.txt", "w")
    for i in sorted(selected_video_numbers):
        video_list.write("file processed/video%s.h264\n" % i)
    video_list.close()

    subprocess.call(["ffmpeg", "-f", "concat", "-i", "tmp/list.txt", output_file_path])

    return rating
