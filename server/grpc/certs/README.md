# gRPC TLS Certificates

Place TLS certificate files here for secure gRPC communication.

## Required files (production)
- `server.key` — Server private key
- `server.crt` — Server certificate
- `ca.crt` — CA certificate (for mutual TLS client verification)

## Generate self-signed certs for development

```bash
# Generate CA
openssl req -x509 -newkey rsa:4096 -keyout ca.key -out ca.crt -days 365 -nodes \
  -subj "/CN=GSS-CA"

# Generate server key + CSR
openssl req -newkey rsa:4096 -keyout server.key -out server.csr -nodes \
  -subj "/CN=localhost"

# Sign server cert with CA
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
  -out server.crt -days 365

# Cleanup CSR
rm server.csr ca.key ca.srl
```

## Environment variables
```bash
GSS_GRPC_TLS_ENABLED=true
GSS_GRPC_TLS_CERT=server/grpc/certs/server.crt
GSS_GRPC_TLS_KEY=server/grpc/certs/server.key
GSS_GRPC_TLS_CA=server/grpc/certs/ca.crt        # optional, for mTLS
```

## Security notes
- **NEVER** commit private keys to git
- Use environment variables or mounted secrets in production
- For Kubernetes: mount certs as secrets, not configmaps
