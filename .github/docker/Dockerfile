# The base is digest-pinned: the image tag is the sha1 of this file, so a
# floating base tag would change the image content under an unchanged tag.
# Refresh the digest deliberately — see docs/developer.rst, "Self-hosted
# end-to-end workflows".
FROM ubuntu:24.04@sha256:fe81560990fae86dc4743202b2cb4d3df98d0bcf949173cecef15ce1127ef852


RUN apt update && \
    apt install -y git software-properties-common && \
    add-apt-repository ppa:dotnet/backports && \
    apt-get update && apt-get install -y build-essential dotnet-sdk-7.0 dotnet-sdk-8.0 dotnet-sdk-9.0 cmake clang && \
    # so we need to clean cache:
    apt-get clean && \ 
    rm -rf /var/lib/apt/lists/*
