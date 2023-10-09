FROM ubuntu

RUN apt-get update
RUN apt-get install python3 -y

RUN mkdir worker

ADD ./ProcessResourceLimiter ./worker/Worker
ADD ./test.py ./test.py

#CMD ["/worker/Worker"]
