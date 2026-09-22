"""Command line entry point.

Run as ``speculative-tool-executor`` or ``python -m speculative_tool_executor``.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path

__all__ = ["main"]

_DEMOS = {
    "speculation": ("speculative_tool_executor.sim.demo", "run_speculation_demo"),
}


def _serve(host: str | None, port: int | None) -> int:
    import uvicorn  # noqa: PLC0415 - only needed when serving

    from speculative_tool_executor.core.app import create_app  # noqa: PLC0415
    from speculative_tool_executor.core.config import get_settings  # noqa: PLC0415
    from speculative_tool_executor.core.logging import configure_logging  # noqa: PLC0415
    from speculative_tool_executor.core.tracing import (  # noqa: PLC0415
        install_global_tracer_provider,
    )

    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)
    app = create_app(settings)
    install_global_tracer_provider(app.state.tracer_provider)
    uvicorn.run(
        app,
        host=host or settings.host,
        port=port or settings.port,
        log_config=None,
        access_log=False,
        server_header=False,
    )
    return 0


def _demo(kit: str, workdir: Path) -> int:
    from speculative_tool_executor.core.logging import configure_logging  # noqa: PLC0415

    configure_logging("WARNING", json_output=True, stream=sys.stderr)
    selected = list(_DEMOS) if kit == "all" else [kit]
    ran = 0
    for name in selected:
        module_name, function_name = _DEMOS[name]
        if importlib.util.find_spec(module_name) is None:
            continue  # kit removed from this repo
        function = getattr(importlib.import_module(module_name), function_name)
        summary = function(workdir / name)
        print(json.dumps({"demo": name, **summary}, indent=2, sort_keys=True))
        ran += 1
    return 0 if ran else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="speculative-tool-executor")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the HTTP service")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    sub.add_parser("generate-api-key", help="print a new API key and its SHA-256 digest")

    audit = sub.add_parser("audit", help="audit log tools")
    audit_sub = audit.add_subparsers(dest="audit_command", required=True)
    verify = audit_sub.add_parser("verify", help="verify a hash-chained audit log")
    verify.add_argument("path", type=Path)
    verify.add_argument("--expect-head", default=None)

    demo = sub.add_parser("demo", help="run the offline demos")
    demo.add_argument("--kit", choices=["all", *sorted(_DEMOS)], default="all")
    demo.add_argument("--workdir", type=Path, default=Path("var/demo"))

    args = parser.parse_args(argv)
    if args.command == "serve":
        return _serve(args.host, args.port)
    if args.command == "generate-api-key":
        from speculative_tool_executor.core.security import generate_api_key  # noqa: PLC0415

        raw, digest = generate_api_key()
        print(json.dumps({"api_key": raw, "api_key_sha256": digest}, indent=2))
        print("Store only api_key_sha256 in APP_API_KEY_SHA256.", file=sys.stderr)
        return 0
    if args.command == "audit":
        from speculative_tool_executor.agentic.audit import main as audit_main  # noqa: PLC0415

        forwarded = ["verify", str(args.path)]
        if args.expect_head:
            forwarded += ["--expect-head", args.expect_head]
        return audit_main(forwarded)
    return _demo(args.kit, args.workdir)


if __name__ == "__main__":
    raise SystemExit(main())
