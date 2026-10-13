#!/usr/bin/env bash
# Build the images, load them into the node's k3s (no registry needed yet),
# and apply the manifests. Images are tagged with the current git commit.
#
# Usage:   deploy/deploy.sh user@node [--import-content]
# Needs:   docker, kubectl pointing at the cluster, SSH to the node, and
#          passwordless sudo for `k3s ctr` on the node (see deploy/README.md).
set -euo pipefail

node=${1:?usage: deploy/deploy.sh user@node [--import-content]}
import_content=${2:-}
cd "$(dirname "$0")/.."

tag=$(git rev-parse --short HEAD)
if [ -n "$(git status --porcelain)" ]; then
  tag="$tag-dirty"
fi
echo "==> deploying $tag to $node"

docker build -f api/Dockerfile -t "forglemmigej-api:$tag" .
docker build -t "forglemmigej-web:$tag" web

for image in forglemmigej-api forglemmigej-web; do
  echo "==> loading $image:$tag into k3s on $node"
  docker save "$image:$tag" | ssh "$node" sudo k3s ctr images import -
done

kubectl kustomize deploy/k8s \
  | sed -E "s/(forglemmigej-(api|web)):latest/\1:$tag/" \
  | kubectl apply -f -
kubectl -n forglemmigej rollout status deployment/api --timeout=180s
kubectl -n forglemmigej rollout status deployment/web --timeout=120s

if [ "$import_content" = "--import-content" ]; then
  echo "==> importing learning content"
  job=$(sed "s/forglemmigej-api:latest/forglemmigej-api:$tag/" deploy/jobs/content-import.yaml \
    | kubectl create -f - -o name)
  kubectl -n forglemmigej wait --for=condition=complete "$job" --timeout=900s
  kubectl -n forglemmigej logs "$job" | tail -3
fi
echo "==> done: $tag"
