# Build context: web/
#   docker build -t forglemmigej-web web
FROM node:24-alpine AS build
WORKDIR /app
COPY package.json package-lock.json .npmrc ./
RUN npm ci
COPY . .
RUN npm run build

# Static files served by nginx; /api is proxied to the API service.
FROM nginxinc/nginx-unprivileged:1.29-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
# A directory of its own: the base image ships a default index.html in
# /usr/share/nginx/html, and the SPA build has no index.html to replace it.
COPY --from=build /app/build /usr/share/nginx/forglemmigej
EXPOSE 8080
