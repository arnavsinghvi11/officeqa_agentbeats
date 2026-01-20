#!/usr/bin/env python3
import json
import os
import subprocess
from datetime import datetime

def get_git_info():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        return {"commit": commit, "branch": branch}
    except Exception:
        return {"commit": "unknown", "branch": "unknown"}

def get_docker_info():
    images = {}
    try:
        result = subprocess.check_output(
            ["docker", "compose", "config", "--images"], text=True
        )
        for line in result.strip().split("\n"):
            if line:
                images[line] = {"pulled_at": datetime.utcnow().isoformat()}
    except Exception:
        pass
    return images

def main():
    provenance = {
        "timestamp": datetime.utcnow().isoformat(),
        "git": get_git_info(),
        "docker_images": get_docker_info(),
        "environment": {
            "github_repository": os.environ.get("GITHUB_REPOSITORY", ""),
            "github_run_id": os.environ.get("GITHUB_RUN_ID", ""),
            "github_actor": os.environ.get("GITHUB_ACTOR", ""),
        },
    }

    with open("provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)

    print("Recorded provenance to provenance.json")

if __name__ == "__main__":
    main()
