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
sudo). Give the machine a DHCP reservation in the router. If the disk still holds an old
Proxmox LVM volume group (`pve`), the installer refuses to reuse it: delete the volume
group under Manual → Configure the Logical Volume Manager first (or run
`vgremove -ff pve` and `wipefs -a /dev/nvme0n1` in the installer shell), then partition.

**Installing where there is no network** (and moving the machine to the router
afterwards, without a monitor): use the full `debian-13.x.x-amd64-DVD-1.iso`, choose
"Do not configure the network at this time" and no network mirror. Before switching
off, still with the monitor attached:

```bash
python3 --version                      # Ansible needs it; if missing: sudo apt install python3 (from the install stick)
ip -br link                            # note the wired interface name (e.g. eno1) and its MAC address
printf 'allow-hotplug eno1\niface eno1 inet dhcp\n' | sudo tee /etc/network/interfaces.d/eno1
```

At the router, find the MAC address in the client list to get the IP (and reserve it).
The playbook replaces the install-medium package source with the online mirrors.

Everything after the installation is the Ansible playbook in `deploy/ansible/`:

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

## MLflow and nightly training

`nightly-training` runs at 19:30 (before the backup and the evening shutdown). It
records a data summary and every model's score on recent reviews in MLflow, and
fits a new FSRS challenger once there are enough reviews. Run it now, or open the
MLflow UI from the laptop:

```bash
kubectl -n forglemmigej create job --from=cronjob/nightly-training train-now
kubectl -n forglemmigej port-forward svc/mlflow 5000:5000      # then http://localhost:5000
```

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
