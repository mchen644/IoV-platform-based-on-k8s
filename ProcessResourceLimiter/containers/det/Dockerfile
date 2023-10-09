#FROM docker.io/luoxiao23333/fusion:v0@sha256:546403b9edb918f9e55ddd6f09d7b42cdfb5673e4c2a6c06884ba85d20e4f18e
#FROM sha256:546403b9edb918f9e55ddd6f09d7b42cdfb5673e4c2a6c06884ba85d20e4f18e

FROM docker.io/luoxiao23333/mcmot:v1

ADD ProcessResourceLimiter /workspace/ProcessResourceLimiter
ADD task_runner_config.yaml /

RUN chmod +x /workspace/ProcessResourceLimiter

CMD ["bash", "-c", "/workspace/ProcessResourceLimiter"]
