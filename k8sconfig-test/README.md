# SimLingo split-agent Kubernetes test

This directory deploys the three SimLingo split agents into the `default`
namespace. Bench2Drive MCP and the NATS broker are intentionally excluded.

## Manifests

- `encoder-agent.yaml`: `simlingo-encoder-agent:0.1.1`
- `scheduler-agent.yaml`: `simlingo-scheduler-agent:0.1.2`
- `llm-agent.yaml`: `simlingo-llm-agent:0.1.1`

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

The agents expect an existing NATS JetStream Service named `nats` in the same
`default` namespace:

```text
nats://nats:4222
```

If the existing NATS Service has a different name, update `NATS_SERVER_URL` in
the three agent manifests.

## Smoke test entry points

After the pods are Ready, trigger the same five-stage flow documented in
`simlingo_agents/SMOKE_TEST.md`, using the service addresses from inside the
cluster or the NodePorts from outside the cluster.
