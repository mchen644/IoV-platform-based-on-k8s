../../buildkit/bin/buildctl build \
    --frontend=dockerfile.v0 \
    --local context=. \
    --local dockerfile=. \
    --output type=image,name=docker.io/chenmin2333/monitor:v0
