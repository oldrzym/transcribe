#!/bin/sh
set -eu

envsubst '${SUMMARY_API_BASE_URL} ${DEFAULT_PROMPT_TEXT}' \
  < /usr/share/nginx/html/config.template.js \
  > /usr/share/nginx/html/config.js
