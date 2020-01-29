docker build \
  --build-arg http_proxy="http://192.168.1.11:8118" \
  --build-arg https_proxy="https://192.168.1.11:8118" \
  --rm \
  -t ti -f Dockerfile .
