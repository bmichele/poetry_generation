#!/usr/bin/env bash

URL=http://0.0.0.0
PORT=8000

curl -v -X GET ${URL}:${PORT}/health
