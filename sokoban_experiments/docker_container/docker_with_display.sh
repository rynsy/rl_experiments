xhost +SI:localuser:ryan
docker run -it --rm -e DISPLAY=:0 -v /tmp/.X11-unix:/tmp/.X11-unix:rw --ipc=host --user 1000:1000 --cap-drop=ALL --security-opt=no-new-privileges sokoban bash
