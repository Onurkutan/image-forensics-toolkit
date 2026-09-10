"""The workbench: HTTP over the service layer, and the client that talks to it.

The client described in ``docs/design/01_toolbox_architecture.md`` (section
3.3) is a thin viewer -- it draws a tool tree, moves sliders, pans one shared
canvas, and downloads a report. Every number it shows and every pixel it
paints has to come from a tool in the registry, or a benchmark table would
stop describing what the user sees. This package is the wire between the two:
:func:`create_app` returns a FastAPI application whose routes are a direct
mapping of :mod:`imgforensics.service` -- the catalogue, a session, a tool
run, a map tile, the fused verdict, the report -- with JSON for the numbers,
PNG for the pixels, and no forensics of its own.

The viewer itself ships here too, under ``static/``: plain HTML, CSS and
ES-module JavaScript, no build step and no CDN (section 3.3, milestone 6d).
``GET /`` returns that page and ``/static`` serves the rest, so one process
and one URL give the whole workbench, a wheel carries it, and it works with
no network beyond the server it came from. :func:`static_dir` says where
those files are, for a deployment that would rather hand them to a web server
than to this one.

What it deliberately is not:

- **Not authenticated.** There are no users, no keys and no rate limits.
  Anything reachable from the internet needs a reverse proxy in front of it
  that provides them; the default bind is loopback for that reason.
- **Not persistent.** Sessions live in one process's memory behind
  :class:`~imgforensics.service.SessionStore`'s TTL and population cap, so a
  restart loses them, and two replicas do not share them. That is the price
  of needing no database, and it is the right price for a demo and for a
  local workbench.
- **Not a place weights come from.** A tool whose weights are absent is
  listed as ``installed: false`` and abstains if it is run, exactly as on the
  CLI. Fetching them stays ``imgforensics weights fetch --accept-license``.

Run it with the optional ``api`` extra installed::

    pip install -e ".[api]"
    imgforensics serve --host 127.0.0.1 --port 8000

``imgforensics serve`` builds the application with
:func:`~imgforensics.api.app.create_app` and hands it to uvicorn; embedding it
in another ASGI server is the same one call. The interactive schema at
``/docs`` is the client contract, generated from the routes themselves.
"""

from imgforensics.api.app import create_app, static_dir

__all__ = ["create_app", "static_dir"]
