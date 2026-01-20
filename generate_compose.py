#!/usr/bin/env python3
import tomli
import yaml
import os

def load_scenario():
    with open("scenario.toml", "rb") as f:
        return tomli.load(f)

def generate_compose(scenario):
    services = {}

    green = scenario.get("green_agent", {})
    green_image = green.get("image", "ghcr.io/OWNER/officeqa-judge:latest")
    green_env = green.get("env", {})

    resolved_env = {}
    for key, value in green_env.items():
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            resolved_env[key] = os.environ.get(env_var, "")
        else:
            resolved_env[key] = value

    services["judge"] = {
        "image": green_image,
        "ports": ["9009:9009"],
        "environment": resolved_env,
        "networks": ["agentnet"],
        "depends_on": ["participant"],
    }

    for i, participant in enumerate(scenario.get("participants", [])):
        p_image = participant.get("image", "ghcr.io/OWNER/officeqa-agent:latest")
        p_env = participant.get("env", {})
        p_name = participant.get("name", f"participant_{i}")

        resolved_p_env = {}
        for key, value in p_env.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                resolved_p_env[key] = os.environ.get(env_var, "")
            else:
                resolved_p_env[key] = value

        services[p_name] = {
            "image": p_image,
            "ports": [f"{9019 + i}:{9019 + i}"],
            "environment": resolved_p_env,
            "networks": ["agentnet"],
        }

    compose = {
        "version": "3.8",
        "services": services,
        "networks": {
            "agentnet": {"driver": "bridge"}
        },
    }

    with open("docker-compose.yml", "w") as f:
        yaml.dump(compose, f, default_flow_style=False)

    print("Generated docker-compose.yml")

if __name__ == "__main__":
    scenario = load_scenario()
    generate_compose(scenario)
