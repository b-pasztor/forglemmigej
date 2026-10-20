#!/usr/bin/env bash
# Run the champion/challenger gate's CLI as a one-off Job in the trainer image
# (same image, secret and settings as the nightly job).
#
# Usage:   deploy/gate.sh report
#          deploy/gate.sh promote VERSION "why"     # e.g. a rollback
set -euo pipefail

case ${1:-} in
  report) args='"--report"' ;;
  promote) args="\"--promote\", \"${2:?version}\", \"--reason\", \"${3:?reason}\"" ;;
  *) echo "usage: deploy/gate.sh report | promote VERSION REASON" >&2; exit 2 ;;
esac

job="gate-$1-$(date +%s)"
kubectl -n forglemmigej create job --from=cronjob/nightly-training "$job" --dry-run=client -o yaml \
  | sed -E "s/^( *)name: train$/&\n\1command: [\"python\", \"-m\", \"forglemmigej_api.training.gate\", $args]/" \
  | kubectl create -f - >/dev/null
kubectl -n forglemmigej wait --for=condition=complete "job/$job" --timeout=600s
kubectl -n forglemmigej logs "job/$job" -c train
kubectl -n forglemmigej delete job "$job" >/dev/null
