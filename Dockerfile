FROM alpine:latest

# Xray install karne ke liye
RUN apk add --no-cache --virtual .build-deps ca-certificates curl \
    && curl -L -H "Cache-Control: no-cache" -o /tmp/xray.zip https://github.com/XTLS/Xray-core/releases/latest/download/Xray-linux-64.zip \
    && unzip /tmp/xray.zip -d /usr/bin \
    && chmod +x /usr/bin/xray \
    && rm -rf /tmp/xray.zip

# Config file copy karein
COPY config.json /etc/xray/config.json

# Render ke dynamic port ke liye entrypoint
CMD ["sh", "-c", "/usr/bin/xray -config /etc/xray/config.json"]
