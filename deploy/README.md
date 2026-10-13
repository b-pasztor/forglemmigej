# Deployment

forglemmigej runs on a single-node k3s cluster (HP ProDesk G4, Debian 13).
Phone access goes through Tailscale; nothing is exposed to the internet.

```
deploy/
  deploy.sh              build images, load them into k3s, apply manifests
  k8s/                   kustomize base: postgres, api (+ migration init container), web, ingress, nightly backup
  jobs/content-import.yaml   one-off job that downloads the sources and imports the deck
```

## 1. Prepare the node (once)

1. Install Debian 13 (minimal, SSH server, no desktop). Give it a fixed IP or a DHCP reservation.
2. Install k3s:
   ```bash
   curl -sfL https://get.k3s.io | sh -
   ```
3. Allow your deploy user to load images without a password prompt (`deploy.sh` pipes images over SSH):
   ```bash
   echo "$USER ALL=(root) NOPASSWD: /usr/local/bin/k3s ctr images import -" | sudo tee /etc/sudoers.d/k3s-import
   ```
4. Copy `/etc/rancher/k3s/k3s.yaml` to your laptop as a kubeconfig and replace `127.0.0.1` with the node's address.

## 2. Create the database secret (once)

The password never goes into git:

```bash
kubectl create namespace forglemmigej
kubectl -n forglemmigej create secret generic forglemmigej-db \
  --from-literal=POSTGRES_DB=forglemmigej \
  --from-literal=POSTGRES_USER=forglemmigej \
  --from-literal=POSTGRES_PASSWORD="$(openssl rand -base64 24)"
```

## 3. Deploy

```bash
deploy/deploy.sh user@node --import-content   # first time: also imports the deck
deploy/deploy.sh user@node                    # later deploys
```

The app is then served by Traefik on `http://<node>/` inside the home network.
Schema migrations run automatically (init container) before each new API version starts.

## Backups

`postgres-backup` runs nightly at 02:30 (Copenhagen time) and keeps 14 days of
`pg_dump` files on the `backups` volume. Test a restore before relying on it:

```bash
kubectl -n forglemmigej create job --from=cronjob/postgres-backup backup-now
```

## Later

- Tailscale (HTTPS on `*.ts.net`, needed for the microphone and the PWA)
- Images built by GitHub Actions and pushed to a registry; GitOps instead of `deploy.sh`
- Copying backups off the node (EliteDesk)
