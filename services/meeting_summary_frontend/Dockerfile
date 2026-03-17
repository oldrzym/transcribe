FROM nginx:1.27-alpine

ENV SUMMARY_API_BASE_URL=http://localhost:8082

RUN apk add --no-cache gettext

COPY public /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY docker-entrypoint.d/40-write-config.sh /docker-entrypoint.d/40-write-config.sh

RUN chmod +x /docker-entrypoint.d/40-write-config.sh

EXPOSE 80
