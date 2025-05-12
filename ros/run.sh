#!/usr/bin/bash

touch .bash_history
docker run -it --rm --privileged \
    --name eye-detector \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="$(pwd)/.bash_history:/root/.bash_history" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="${XAUTHORITY}:/root/.Xauthority" \
    --volume="$(pwd)/ws:/ws" \
    --volume="$(pwd)/../src:/ws/src_py/src" \
    --volume="$(pwd)/../outdata:/ws/outdata" \
    --volume="$(pwd)/../indata:/ws/indata" \
    --volume="$(pwd)/../middata:/ws/middata" \
    eye-detector
