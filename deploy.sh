scp away.py config.yml motion_detector.py requirements.txt video_composer.py video_processor.py pi@pi:~
ssh pi@pi 'pip3 install -r requirements.txt; sudo stop away; sudo start away'
