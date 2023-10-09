ctr -n buildkit i rm docker.io/luoxiao23333/task_det:v0

/home/hanhan/git-repo/buildkit/bin/buildctl build \
    --frontend=dockerfile.v0 \
    --local context=. \
    --local dockerfile=. \
    --no-cache \
    --output type=image,name=docker.io/luoxiao23333/task_det:v0
