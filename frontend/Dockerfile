# ============================================================
# KrishiRakshak Frontend Dockerfile
# Multi-stage: build with Node, serve with Nginx
# ============================================================

# ---- Build stage ----
FROM node:20-alpine AS builder

WORKDIR /app

# Build args (can be overridden in docker-compose)
ARG REACT_APP_API_URL=/api
ENV REACT_APP_API_URL=$REACT_APP_API_URL

# Install dependencies (cached layer)
COPY package.json ./
RUN npm install --legacy-peer-deps

# Copy source and build
COPY . .
RUN npm run build

# ---- Serve stage ----
FROM nginx:alpine AS runner

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Add custom nginx config with API proxy
COPY --from=builder /app/build /usr/share/nginx/html

# Nginx config: serve React SPA + proxy /api to backend
RUN cat > /etc/nginx/conf.d/krishirakshak.conf << 'EOF'
server {
    listen 80;
    server_name _;
    root /usr/share/nginx/html;
    index index.html;

    # React SPA routing
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Proxy API requests to backend
    location /api/ {
        proxy_pass http://backend:8000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 120s;
        proxy_connect_timeout 10s;
    }

    # Gzip compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml;
    gzip_min_length 1024;
}
EOF

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
