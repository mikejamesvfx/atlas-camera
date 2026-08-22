"""Drive Adobe Photoshop (Beta) as an Atlas paint bridge.

The only Photoshop-specific tool in the set: everything downstream of the paint
package (confine, score, ROI) is vendor-neutral and lives in
``atlas_camera.paint``.

Rungs, strictest first — each is a separate flag so a failure is attributable::

    python tools/photoshop_bridge.py --probe
        Rung C. Connect, verify the install path, report version and OCIO state.
        Touches no document.

    python tools/photoshop_bridge.py --open <plate>.exr [--no-ocio]
        Rung B. Open as an OpenColorIO document and report the mode and bit
        depth it actually opened at. A plate that quietly opened at 16-bit
        means the OCIO leg never engaged.

    python tools/photoshop_bridge.py --open <plate>.exr --fill "<prompt>" \
        --export <out>.exr [--fill-mode inpaint|variation|synthesize]
        Rung A. The generative leg. The RAW return is what gets scored, before
        any Atlas-side confinement, because that is what measures the vendor's
        own containment behaviour.

Always pass ``--ocio-config`` (or set ``$OCIO``) so both applications resolve
the same config. A colourspace name without a config is not a contract.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main(argv: list[str] | None = None) -> int:
    from atlas_camera.paint import ocio, vendors
    from atlas_camera.paint.photoshop.com_client import (DEFAULT_INSTALL_DIR,
                                                         PhotoshopBridgeError,
                                                         PhotoshopClient)

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--probe", action="store_true",
                   help="Rung C: connect and identify, touching no document")
    p.add_argument("--open", dest="open_path", type=Path, default=None)
    p.add_argument("--no-ocio", action="store_true",
                   help="Open WITHOUT OpenColorIO (the 8/16-bit generative lane)")
    p.add_argument("--working-space", default="ACEScg")
    p.add_argument("--ocio-domain", default="Environment",
                   help="OCIO config domain. 'Environment' makes Photoshop read "
                        "$OCIO, which is how it lands on the same config as Atlas")
    p.add_argument("--convert", action="store_true",
                   help="Run convertToOCIO after opening")
    p.add_argument("--fill", default=None, metavar="PROMPT",
                   help="Rung A: run Generative Fill on the current selection")
    p.add_argument("--fill-mode", default="inpaint",
                   choices=("inpaint", "variation", "synthesize"))
    p.add_argument("--export", dest="export_path", type=Path, default=None)
    p.add_argument("--install-dir", default=DEFAULT_INSTALL_DIR)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--scratch", type=Path, default=None)
    p.add_argument("--ocio-config", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None,
                   help="Write the collected results as JSON")
    args = p.parse_args(argv)

    if not any((args.probe, args.open_path, args.fill, args.export_path)):
        p.error("nothing to do: pass --probe, --open, --fill or --export")

    results: dict = {"vendor": vendors.PHOTOSHOP_BETA.key}
    for caveat in vendors.PHOTOSHOP_BETA.caveats:
        print(f"note: {caveat}", file=sys.stderr)

    with ocio.scoped_config(args.ocio_config) as identity:
        print(ocio.describe(identity), file=sys.stderr)
        results["ocio"] = identity

        client = PhotoshopClient(install_dir=args.install_dir,
                                 scratch_dir=args.scratch,
                                 timeout_s=args.timeout)
        try:
            results["app"] = client.connect()
            print(f"connected: {results['app'].get('name')} "
                  f"{results['app'].get('version')}")
            print(f"install:   {results['app'].get('path')}")
            print(f"OCIO seen by Photoshop: "
                  f"{results['app'].get('ocioEnv') or '(unset)'}")

            if args.open_path:
                opened = client.open_plate(args.open_path,
                                           as_ocio=not args.no_ocio)
                results["open"] = opened
                print(f"opened:    {opened.get('name')} "
                      f"{opened.get('width')}x{opened.get('height')} "
                      f"{opened.get('bitsPerChannel')} {opened.get('mode')}")
                # An OCIO open that silently landed at 16-bit means the colour
                # leg never engaged, and every number measured after it would
                # be meaningless. Say so rather than continuing quietly.
                if not args.no_ocio and "THIRTYTWO" not in str(
                        opened.get("bitsPerChannel", "")).upper():
                    print("WARNING: asked for an OpenColorIO document but it did "
                          "not open at 32 bits/channel — the OCIO leg did not "
                          "engage. Treat any colour measurement from this run as "
                          "unfounded.", file=sys.stderr)

            if args.convert:
                results["convert"] = client.convert_to_ocio(
                    working_space=args.working_space,
                    configuration=args.ocio_domain)
                print(f"converted: working space {args.working_space} "
                      f"({args.ocio_domain})")

            if args.fill:
                results["fill"] = client.generative_fill(
                    prompt=args.fill, mode=args.fill_mode)
                print(f"filled:    mode={args.fill_mode} prompt={args.fill!r}")

            if args.export_path:
                results["export"] = client.export(args.export_path)
                print(f"exported:  {args.export_path}")

        except PhotoshopBridgeError as exc:
            print(f"\nBRIDGE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            results["error"] = {"type": type(exc).__name__, "message": str(exc)}
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(results, indent=2, sort_keys=True),
                                    encoding="utf-8")
            return 1

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, sort_keys=True),
                            encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
