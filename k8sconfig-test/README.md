# SimLingo split-agent Kubernetes test

This directory deploys the three SimLingo split agents into the `default`
namespace. Bench2Drive MCP and the NATS broker are intentionally excluded.

## Manifests

- `encoder-agent.yaml`: `simlingo-encoder-agent:0.2.5`
- `scheduler-agent.yaml`: `simlingo-scheduler-agent:0.2.5`
- `llm-agent.yaml`: `simlingo-llm-agent:0.2.5`

Each agent requests and limits one GPU:

```yaml
resources:
  requests:
    nvidia.com/gpu: "1"
  limits:
    nvidia.com/gpu: "1"
```

## Deploy

Make sure the listed images are available to the Kubernetes nodes. If the
cluster cannot see local Docker images directly, push the images to a registry
and update the `image:` fields first.

```bash
kubectl apply -f k8sconfig-test/encoder-agent.yaml
kubectl apply -f k8sconfig-test/scheduler-agent.yaml
kubectl apply -f k8sconfig-test/llm-agent.yaml
```

Check status:

```bash
kubectl get pods,svc -n default
```

## Service ports

- encoder HTTP: service port `9011`, nodePort `30111`
- llm HTTP: service port `9012`, nodePort `30112`
- scheduler HTTP: service port `9013`, nodePort `30113`

The agents expect an existing edge NATS JetStream Service and an
`edge-cluster-config` ConfigMap in the same `default` namespace:

```text
NATS_SERVERS=nats://nats:4222
CLUSTER_ID=edge-c
NATS_JETSTREAM_DOMAIN=edge-c
```

Each Pod receives `AGENT_INSTANCE_ID=metadata.uid` and manages its own
`WF_<pod-uid>` WorkQueue Stream. Full input Subjects are expanded from the
cluster, agent and Pod identity. Cross-agent output Subjects are supplied by
the evaluation client in `/a2a/execute` metadata.

After all Pods are Ready, record their UIDs:

```bash
ENCODER_UID="$(kubectl -n default get pod -l app=simlingo-encoder-agent \
  -o jsonpath='{.items[0].metadata.uid}')"
SCHEDULER_UID="$(kubectl -n default get pod -l app=simlingo-scheduler-agent \
  -o jsonpath='{.items[0].metadata.uid}')"
LLM_UID="$(kubectl -n default get pod -l app=simlingo-llm-agent \
  -o jsonpath='{.items[0].metadata.uid}')"
```

Pass these values to `start_eval_split_agents_local.py`. Pod UIDs change after
a rollout, so they must be queried again before a later evaluation.

## Smoke test entry points

After the pods are Ready, trigger the same five-stage flow documented in
`simlingo_agents/SMOKE_TEST.md`, using the service addresses from inside the
cluster or the NodePorts from outside the cluster.
