/home/hanhan/git-repo/buildkit/bin/buildctl build \
    --frontend=dockerfile.v0 \
    --local context=. \
    --local dockerfile=. \
    --output type=image,name=docker.io/luoxiao23333/task_mcmot:v0
