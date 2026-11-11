import ipaddress
import json

import pytest

from flock.harbor import networking


def test_subnets_exclude_vpn_routes_from_all_tables():
    routes = [
        {"dst": "default", "gateway": "192.168.1.1"},
        {"dst": "172.17.0.0/16", "dev": "docker0"},
        {"dst": "172.18.0.0/16", "dev": "tailscale0", "table": 52},
        {"dst": "172.20.0.0/17", "dev": "tailscale0", "table": 52},
        {"dst": "172.30.1.9", "type": "local"},
    ]
    candidates = networking.candidate_subnets(routes)
    assert candidates
    assert all(n.prefixlen == 24 for n in candidates)
    for route in routes[1:]:
        occupied = ipaddress.ip_network(route["dst"])
        assert all(not n.overlaps(occupied) for n in candidates)


@pytest.mark.parametrize("engine", ["docker", "podman"])
async def test_allocator_handles_concurrent_reservation_and_owns_cleanup(
    tmp_path, monkeypatch, engine
):
    calls = []
    creates = 0

    async def command(*args):
        nonlocal creates
        calls.append(args)
        if args[0] == "ip":
            return 0, "[]", ""
        if args[2] == "create":
            creates += 1
            return (
                (1, "", "Pool overlaps with other one on this address space")
                if creates == 1
                else (0, "network-id", "")
            )
        if args[2] == "inspect":
            return (
                0,
                json.dumps(
                    [
                        {
                            ("labels" if engine == "podman" else "Labels"): {
                                "io.flock.run": "test-run"
                            }
                        }
                    ]
                ),
                "",
            )
        return 0, "", ""

    monkeypatch.setattr(networking, "command", command)
    networks = networking.TrialNetworks("test-run", engine=engine)
    await networks.start()
    path = await networks.create("agent-001", tmp_path / "overlay.json")
    assert creates == 2
    assert json.loads(path.read_text())["networks"]["default"] == {
        "external": True,
        "name": "flock-test-run-agent-001",
    }
    await networks.stop()
    assert calls[-1] == (engine, "network", "rm", "flock-test-run-agent-001")


@pytest.mark.parametrize("failure", ["ownership", "daemon"])
async def test_cleanup_reports_failures_and_never_deletes_foreign_networks(
    monkeypatch, failure
):
    calls = []

    async def command(*args):
        calls.append(args)
        if failure == "daemon":
            return 1, "", "Cannot connect to the Docker daemon"
        return 0, json.dumps([{"Labels": {"io.flock.run": "another-run"}}]), ""

    monkeypatch.setattr(networking, "command", command)
    networks = networking.TrialNetworks("test-run")
    networks.names = ["a-network"]
    with pytest.raises(RuntimeError):
        await networks.stop()
    assert all(call[2] == "inspect" for call in calls)


async def test_podman_ignores_unrelated_docker_endpoint(monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "ssh://unrelated-host")
    monkeypatch.delenv("CONTAINER_HOST", raising=False)

    async def command(*args):
        return 0, "[]", ""

    monkeypatch.setattr(networking, "command", command)
    await networking.TrialNetworks("local", engine="podman").start()
    monkeypatch.setenv("CONTAINER_HOST", "ssh://remote-host")
    with pytest.raises(ValueError, match="local Linux"):
        await networking.TrialNetworks("remote", engine="podman").start()
