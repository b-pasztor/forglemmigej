# Deployment

forglemmigej runs on a single-node k3s cluster (HP ProDesk G4, Debian 13).
Phone access goes through Tailscale; nothing is exposed to the internet.

```
deploy/
  ansible/               node setup: Debian 13 -> Tailscale + k3s (site.yml)
  deploy.sh              build images, load them into k3s, apply manifests
  k8s/                   kustomize base: postgres, api (+ migration init container), web, ingress, daily backup
  jobs/content-import.yaml   one-off job that downloads the sources and imports the deck
```

## 1. Prepare the node (once)

Install Debian 13 by hand (netinst, no desktop; software selection: only "SSH server"
and "standard system utilities"; leave the root password empty so your user gets
sudo). Give the machine a DHCP reservation in the router. Everything after that is
the Ansible playbook in `deploy/ansible/`:

- base packages, timezone, unattended security upgrades
- Tailscale, logged in with a one-time auth key, and `tailscale serve` for HTTPS on `https://<node>.<tailnet>.ts.net`
- single-node k3s with graceful node shutdown (PostgreSQL stops cleanly when the machine is switched off) and the Tailscale name in the API certificate
- a sudo rule so `deploy.sh` can load images into k3s
- a kubeconfig for the laptop (`deploy/ansible/kubeconfig`, gitignored), pointing at the node's Tailscale name

Ansible's control side does not run on Windows, so run it from WSL:

```bash
sudo apt install -y ansible                                  # once, in WSL
ssh-copy-id deploy@192.168.1.50                              # WSL has its own SSH key
cd /mnt/d/Development/MLOps/deploy/ansible
cp inventory.example.ini inventory.ini                       # then set the address and user
TAILSCALE_AUTHKEY=tskey-auth-... ansible-playbook -i inventory.ini site.yml -K
```

The auth key comes from the Tailscale admin console (Settings → Keys → Generate auth
key; one-off is enough). `-K` asks for your sudo password on the node. Re-running the
playbook is safe.

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

The app is then served on `https://<node>.<tailnet>.ts.net/` to devices on your tailnet
(Tailscale terminates HTTPS and forwards to Traefik on port 80).
Schema migrations run automatically (init container) before each new API version starts.

## Backups

`postgres-backup` runs daily at 20:00 (Zurich time), before the node is switched off
for the night; a run missed while the node was off starts at the next boot. It keeps
14 days of `pg_dump` files on the `backups` volume. Test a restore before relying on it:

```bash
kubectl -n forglemmigej create job --from=cronjob/postgres-backup backup-now
```

## Later

- Images built by GitHub Actions and pushed to a registry; GitOps instead of `deploy.sh`
- Copying backups off the node (EliteDesk)
