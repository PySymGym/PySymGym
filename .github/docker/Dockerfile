FROM ubuntu:24.04


RUN apt update && \
    apt install -y git software-properties-common && \
    add-apt-repository ppa:dotnet/backports && \
    apt-get update && apt-get install -y build-essential dotnet-sdk-7.0 dotnet-sdk-9.0 cmake clang maven && \
    # so we need to clean cache:
    apt-get clean && \ 
    rm -rf /var/lib/apt/lists/*
