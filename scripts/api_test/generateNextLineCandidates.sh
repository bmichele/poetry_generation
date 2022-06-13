#!/usr/bin/env bash

URL=http://0.0.0.0
PORT=8000

curl -v -H "key: supersecret" -H "Content-Type: application/json" -X POST "${URL}:${PORT}/generator/next_line" \
  -d '{
        "poem_id": "9cd64370-d30f-4d9e-98d6-7f41cf5d780d",
        "line_id": 1,
        "language": "en",
        "style": "",
        "poem_state":["The test and the example of mankind,","To guard the virtues of their noblest minds,"]
      }'
