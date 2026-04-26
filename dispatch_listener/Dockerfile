ARG BUILD_FROM
FROM ${BUILD_FROM}

ENV LANG=C.UTF-8 PYTHONUNBUFFERED=1

# Audio + Python deps
RUN apk add --no-cache \
      python3 py3-pip py3-numpy \
      pulseaudio-utils alsa-lib \
      tini

# App deps
COPY service/requirements.txt /opt/dispatch_listener/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages \
      -r /opt/dispatch_listener/requirements.txt

# App code
COPY service/ /opt/dispatch_listener/

# s6 service script (runs at container start)
COPY rootfs/ /

ENTRYPOINT ["/sbin/tini", "--"]
CMD ["/etc/services.d/dispatch_listener/run"]
