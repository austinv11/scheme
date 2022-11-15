#!/usr/bin/env sh

docker1 stop aav4003__biohpc_scheme
docker1 rm aav4003__biohpc_scheme


# Check if first argument exists and is gpu
if [ "$1" = "gpu" ]; then
    docker_cmd_prefix="docker1 run --gpus all"
else
    docker_cmd_prefix="docker1 run"
fi

$docker_cmd_prefix -d --name scheme -p 8033:41502 -p 8034:64456 -p 8039:8888 -p 8035:22 -v /workdir/aav4003/scheme:/mounted/scheme biohpc_aav4003/scheme:latest