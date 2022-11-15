#!/usr/bin/env sh

SSH_SERVER="$1"  # bash ./run_docker_in_server.sh cwid@cbsugpu02

# Upload the docker run script and launch it
echo "Uploading data (this will take awhile)..."

docker1 save biohpc_aav4003/scheme:latest > scheme.tar

rsync -v -r ../scheme "$SSH_SERVER":/workdir/aav4003/

rm -f scheme.tar
echo "Launching docker..."
ssh "$SSH_SERVER" "cd /workdir/aav4003/scheme && docker1 load -i scheme.tar && rm -f scheme.tar && bash ./run_docker.sh gpu"