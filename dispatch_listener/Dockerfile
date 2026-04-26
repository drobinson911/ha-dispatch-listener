ARG BUILD_FROM
FROM ${BUILD_FROM}

ENV LANG=C.UTF-8 PYTHONUNBUFFERED=1

# Runtime audio + Python
RUN apk add --no-cache \
      python3 py3-pip py3-numpy \
      pulseaudio-utils alsa-lib \
      ffmpeg \
      tini

# Build deps for pywhispercpp (compiles whisper.cpp from source)
RUN apk add --no-cache --virtual .build-deps \
      build-base cmake git python3-dev

COPY service/requirements.txt /opt/dispatch_listener/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages \
      -r /opt/dispatch_listener/requirements.txt \
    && apk del .build-deps

COPY service/ /opt/dispatch_listener/
COPY rootfs/ /

ENTRYPOINT ["/sbin/tini", "--"]
CMD ["/etc/services.d/dispatch_listener/run"]
