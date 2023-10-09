FROM docker.io/mchen644/mcmot:v0

ADD ProcessResourceLimiter ProcessResourceLimiter
ADD task_runner_config.yaml task_runner_config.yaml

RUN chmod +x ProcessResourceLimiter

CMD ["bash", "-c", "./ProcessResourceLimiter"]
