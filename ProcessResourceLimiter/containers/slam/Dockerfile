FROM ubuntu

ADD ./Pangolin/ Pangolin
ADD build_pangolin.sh build_pangolin.sh
RUN bash build_pangolin.sh

ADD ./opencv3/ opencv3
ADD build_opencv3.sh build_opencv3.sh
RUN bash build_opencv3.sh

ADD ./ORB_SLAM2 ORB_SLAM2
ADD build_slam.sh build_slam.sh
RUN bash build_slam.sh

ADD ProcessResourceLimiter /ORB_SLAM2/ProcessResourceLimiter
ADD task_runner_config.yaml task_runner_config.yaml

RUN chmod +x /ORB_SLAM2/ProcessResourceLimiter

CMD ["bash", "-c", "/ORB_SLAM2/ProcessResourceLimiter"]