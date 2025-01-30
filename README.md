# Setup

## With Podman

### Dependencies
1. Podman with nvidia toolkit enabled
2. Python
3. Access to qluvio repo via ssh key

#### Download models
1. `./pull-models`
2. Make sure config.yml:storage/system/model_path points to <models>/logo, where <models> is the output of the above. 

#### Add ssh keys to ssh-agent
`ssh-add` (on personal machine)

**NOTE**: if you are on a remote server, either you should have your ssh key on the remote server and run `ssh-add` there, or you should run it on your personal machine and verify that you are connected with agent forwarding enabled.

#### Build image
`./build.sh`

#### Default run
```
podman run --rm --volume=$(pwd)/test:/elv/test:ro --volume=$(pwd)/tags:/elv/tags --volume=$(pwd)/.cache:/root/.cache --network host --device nvidia.com/gpu=0 logo test/1.mp4 test/2.mp4
```

1. Note: you must mount the files to tag into the container storage (`--volume=$(pwd)/test:/elv/test`)
2. Tag files will appear in the `tags` directory (`--volume=$(pwd)/tags:/elv/tags`). 

#### Custom run

1. Default parameters are found in config.yml under `runtime/default`
2. These values can be overriden by passing in data with `--config` when running the container in the command line:

```
podman run --rm --volume=$(pwd)/test:/elv/test:ro --volume=$(pwd)/tags:/elv/tags --volume=$(pwd)/.cache:/root/.cache --network host --device nvidia.com/gpu=0 logo test/1.mp4 test/2.mp4 --config '{"fps":8, "thres":0.6}'
```