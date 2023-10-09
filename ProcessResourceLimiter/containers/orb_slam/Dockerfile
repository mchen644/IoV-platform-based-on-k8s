FROM docker.io/mchen644/orb_slam:v0

ADD ProcessResourceLimiter /ORB_SLAM2/ProcessResourceLimiter
ADD task_runner_config.yaml task_runner_config.yaml

RUN chmod +x /ORB_SLAM2/ProcessResourceLimiter

CMD ["bash", "-c", "/ORB_SLAM2/ProcessResourceLimiter"]
