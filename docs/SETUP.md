# Setup guide: run Towers as Rain Gauges on any machine

There are three ways to run it. Pick one:

| Way | You need | Time | You get |
|---|---|---|---|
| [A. Just look](#a-just-look-no-install) | a web browser | 0 min | the control room, replays and explainer (no live watch) |
| [B. Docker](#b-docker-any-os-easiest-full-install) | Docker | ~15 min | everything, including the live watch |
| [C. Python](#c-python-linux-macos-windows) | Python 3.10–3.12 | ~15 min | everything, and you can rerun the experiments |

Everything runs on a normal computer; a GPU is **not** needed for the pages, the replays, or the live watch.
Three Algerian cities come ready to watch (Béjaïa, Constantine, Tébessa): on a first start without a GPU they are
live within about 2 minutes, then each 5-minute update takes about 25 seconds on 8 CPU cores (5 s on a GPU). A
drill (a made-up storm sent through the chain) takes about 11 minutes on 8 CPU cores, under a minute on a GPU.
The one slow step without a GPU is **adding a new city**: it runs up to 14 flood simulations to find the rain that
would flood the town, which takes a few minutes on a GPU but **several hours on a CPU**. To add many cities, use a
machine with an NVIDIA GPU.

---

## A. Just look (no install)

Open the hosted copy: **https://abd-elhakim-arabet.github.io/towers-as-rain-gauges/**

It has the control room (Algiers 2001 and Ghardaïa 2008 replays), the storm replay, the flood-risk page and the
explainer PDF. The live watch needs a running Python server, so on the hosted copy it only explains how to start one.

---

## B. Docker (any OS, easiest full install)

1. Install Docker: [Docker Desktop](https://www.docker.com/products/docker-desktop/) on Windows and macOS,
   or `docker.io` from your package manager on Linux. On Windows and macOS, start Docker Desktop and wait until it
   says it is running before the next steps.
2. Get the code and build the image (5–10 min the first time, mostly downloading Python packages):

   ```bash
   git clone https://github.com/Abd-Elhakim-Arabet/towers-as-rain-gauges.git
   cd towers-as-rain-gauges
   docker build -t rain-towers .
   ```

3. Start it:

   ```bash
   docker run -p 127.0.0.1:8000:8000 -v rain-towers-data:/app/data rain-towers
   ```

   The first start downloads the demo data (about 300 MB). When it prints `serving on http://0.0.0.0:8000`,
   open **http://localhost:8000**.

The volume `rain-towers-data` keeps the demo data and the cities you watch, so later starts take seconds. Stop with
`Ctrl+C`. Works the same on Apple Silicon and Intel Macs, on Windows (Docker Desktop with WSL 2) and on Linux.
`127.0.0.1:` keeps the page on your own computer; leave it out (`-p 8000:8000`) only if other machines on your network
should open it, because anyone who can open the page can change where alarm emails go.

---

## C. Python (Linux, macOS, Windows)

### 1. Install the tools

- **Python 3.12** (3.10 and 3.11 work too). **Not 3.13 or newer yet:** some map libraries have no ready-made
  install for them, and pip fails. Check with `python3.12 --version` (Windows: `py -3.12 --version`). To get it:
  - Windows: the 3.12 installer from [python.org](https://www.python.org/downloads/windows/); tick "Add Python to PATH".
  - macOS (Apple Silicon): `brew install python@3.12`, or the 3.12 installer from python.org.
  - macOS (Intel): use [Docker](#b-docker-any-os-easiest-full-install) instead. PyTorch no longer publishes
    Intel Mac versions that work with the other libraries.
  - Linux: `sudo apt install python3.12 python3.12-venv` (Ubuntu 24.04, Debian 13), or, on any distribution,
    install [uv](https://docs.astral.sh/uv/) and use `uv venv --python 3.12 .venv` in step 2.
- **git** ([git-scm.com](https://git-scm.com/downloads)).

### 2. Get the code and make an environment

Linux and macOS:

```bash
git clone https://github.com/Abd-Elhakim-Arabet/towers-as-rain-gauges.git
cd towers-as-rain-gauges
python3.12 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
git clone https://github.com/Abd-Elhakim-Arabet/towers-as-rain-gauges.git
cd towers-as-rain-gauges
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell says "running scripts is disabled on this system", run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, answer Y, and activate again.

From here on the commands are the same everywhere (`python` means the one inside `.venv`).

### 3. Install PyTorch, then the rest

Pick the line for your machine:

| Machine | Command |
|---|---|
| Any computer, no GPU (enough for the demo) | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| NVIDIA GPU (Linux or Windows) | `pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| Mac (Intel or Apple Silicon) | `pip install torch` |

Then:

```bash
pip install -r requirements-demo.txt
pip install -e .
```

`requirements-demo.txt` is everything the pages and the live watch need. To rerun the experiments, install
`requirements.txt` instead (it adds `pysteps` and a few research tools; on Python 3.12 `pysteps` is compiled during
install, so you need a C compiler: `build-essential` on Linux, Xcode command-line tools on macOS, or
"Build Tools for Visual Studio" on Windows).

### 4. Get the demo data

The trained models and the replay data are too big for git, so they live in a release file (about 300 MB):

```bash
python scripts/demo_bundle.py get
```

### 5. Check and start

```bash
python -m pytest tests/test_watch.py tests/test_floodrisk.py   # about a minute; all should pass
python scripts/serve_site.py
```

Open **http://localhost:8000**. Stop with `Ctrl+C`.

---

## What you can open

| Address | What it is |
|---|---|
| `/` | start page with links to everything |
| `/dashboard/` | control room: Algiers 2001 and Ghardaïa 2008 replayed, with a guided tour |
| `/watch/` | live watch: type any city, watch its rain and flood risk now, run a drill |
| `/replay/`, `/flood/`, `/storm3d/` | the earlier pages (storm replay, flood risk, 3D storm) |
| `/explainer.pdf` | the whole project in plain language, with every result |

The maps load imagery and terrain from the internet, so keep the machine online.

## Live watch settings (optional)

**Alarm email.** The first time you open `/watch/`, it asks where alarms should go and which account sends them
(a Gmail address with an [app password](https://myaccount.google.com/apppasswords) works best), and has a
"Send a test" button. Change it later with "Alarm email" on the left. It is saved in `data/live/mail.json` on the
server (owner-only, kept in the Docker volume). The password is never shown again. Without a sending account,
alarm emails are saved as files in `data/live/outbox/` instead.

Who can change it: with Python, anyone who opens the page while no address is set yet, and after that only someone
on the computer that runs the server. With Docker, anyone who can open the page (inside Docker every browser looks like
another machine), which is why the `docker run` command above publishes the page on `127.0.0.1` only.

The same settings can also go in a `.env` file (copy `.env.example`; the page's settings win over it, and
environment variables win over both):

- `ALERT_TO`: the address that receives alarm emails.
- `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`: the account that sends them.
- `EARTHDATA_TOKEN` (free, [NASA Earthdata](https://urs.earthdata.nasa.gov/)): better soil data for new cities.
  Without it, soil is assumed average (marked on the page).

To plug in real operator data instead of the simulation, see [adapters/README.md](../adapters/README.md).

## Open it to other machines

```bash
python scripts/serve_site.py --host 0.0.0.0 --port 8000
```

Anyone on the same network can then open `http://<this machine's IP>:8000`. Put it behind a firewall or a reverse
proxy (e.g. nginx with HTTPS) before opening it to the internet.

## Rerun the experiments (reproduce the results)

The demo data above is enough to look and to run the live watch. To rebuild every number in the explainer from the
raw data, follow [REPRODUCE.md](REPRODUCE.md) step by step. You will need:

- about 60 GB of disk for the raw datasets (`python scripts/download_data.py` fetches them; see [DATA.md](DATA.md)
  for sources, licences and sizes);
- free API keys in `.env` for some sources: `OPENCELLID_TOKEN` (tower sites), `EARTHDATA_TOKEN` (NASA satellite
  rain), `KNMI_API_KEY` (Dutch radar);
- a GPU for comfortable training times (CPU works, slower).

The MindSpore version of the model lives in a separate environment (`requirements-mindspore.txt`, needs `numpy<2`).

## Troubleshooting

| Problem | Fix |
|---|---|
| `pip install` fails on `rasterio`, `netcdf4` or `h5py` (or asks for a compiler) | Your Python is too new (3.13 or later) or too old. Check `python --version` inside the environment; make a new one with Python 3.12 (step 1), or use Docker. |
| `CERTIFICATE_VERIFY_FAILED` when getting the demo data (macOS) | Python from python.org needs its certificates: run "Install Certificates.command" in its folder under Applications, then try again. |
| A download stops halfway | Run the same command again; a broken file is detected and fetched again. |
| `pip install -r requirements.txt` fails building `pysteps` | Install a C compiler (see step 3), or use `requirements-demo.txt` if you only need the demo. |
| The dashboard says data is missing | Run `python scripts/demo_bundle.py get`. |
| The live watch page says no server is attached | You opened the static copy. Start `python scripts/serve_site.py` and open `http://localhost:8000/watch/`. |
| Docker: `PermissionError: ... /app/data/demo-bundle.tar` | Fixed on 25 Nov 2026. Run `git pull`, then `docker build -t rain-towers .` again; the existing volume is repaired on the next start. |
| Port 8000 is busy | `python scripts/serve_site.py --port 8080` (Docker: `-p 127.0.0.1:8080:8000`). |
| Maps stay blank | The machine needs internet for map tiles; check a firewall or proxy. |
| A new city takes long to set up | Normal the first time: it downloads terrain and land cover, then tries many storms to find the flood threshold (a few minutes on a GPU, possibly hours on a CPU). Restarts reuse all of it. |
