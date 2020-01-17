#!/bin/bash
for i in $(seq -f '%04g' 0000 0012)
do
    echo ${i}

    rgb_before_raw="${i}_color.jpg"
    depth_before_raw="${i}_depth_colored.png"

    scp xupengfei@192.168.0.117:~/mqa/$rgb_before_raw rgb_before/$rgb_before_raw
    scp xupengfei@192.168.0.117:~/mqa/$depth_before_raw depth_before/$depth_before_raw

    python3 output_action_map.py -scene_index $i
done