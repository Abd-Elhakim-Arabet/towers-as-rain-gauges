# Towers as Rain Gauges

**Flash-flood early warning from the microwave links that already connect cell towers.**

Rain weakens the radio signal between two towers, and mobile operators already record that signal every few
minutes. So every link in a phone network is an accidental rain gauge, and a country covered by towers already has
thousands of them. This project turns those signals into a live rain map with AI, runs a flood model on that rain,
and raises an alarm for the streets most likely to flood in the next hours.

It was built for the Huawei ICT Competition (Innovation Track) with flash floods in Algeria in mind, such as
Bab El Oued (2001) and Ghardaïa (2008), where rain gauges and weather radar are sparse but phone towers are everywhere.

| | |
|---|---|
| **Try it** | https://abd-elhakim-arabet.github.io/towers-as-rain-gauges/ (control room, storm replays, explainer) |
| **Run it yourself** | [docs/SETUP.md](docs/SETUP.md): Docker or Python, any OS, about 15 minutes, includes the live watch |
| **Demo videos** | [1. Control room](https://drive.google.com/file/d/1zzyxVfqiL8HyYQ1QinNahxSd-7Gh3xkk/view) (3.5 min) · [2. Live watch](https://drive.google.com/file/d/1AcTm7JRbZCaxnafX0Fv6DpbN1PAbq2Pw/view) (5 min) · [folder](https://drive.google.com/drive/folders/1aentwt-Wob2NCwFE_o6EMXQmLXrTyCtE) on Google Drive |
| **The whole story** | [explainer PDF](docs/explainer/explainer.pdf): every step and result in plain language |

## How it works

```
tower-to-tower signal levels  →  AI: is it raining on this link, and how hard?
                              →  rain map over the town, every 5 minutes
                              →  flood model: where the water goes in the next 2 hours
                              →  alert level (normal, watch, flood warning, flooding) on screen and by email
```

- **AI rain reader.** A small neural network reads each link's signal and estimates the rain on it. It runs in
  PyTorch, with a MindSpore twin for the Huawei stack.
- **Rain map.** The links' readings are spread into a map of the town.
- **Flood model.** Our own 2D water-flow model (`src/floodrisk/`) moves that rain over the real terrain and land
  cover, and finds the neighbourhoods and roads that flood.
- **Live watch.** Type any city in the world. The system fetches its terrain, finds its towers, builds the links,
  then follows the rain and the flood risk live and sends an alarm email when it matters.

## What is real and what is simulated

- **Real data, real tests (Europe).** The AI was trained and checked on real link recordings from Sweden, Italy and
  the Netherlands, against rain gauges and weather radar. On Italian links it had never seen, its hourly rain
  matched the radar far better than the classical method (correlation 0.64 against 0.22).
- **Simulation (Algeria).** No Algerian operator data was available. The Algerian part uses real tower positions
  and real satellite rainfall to drive a *virtual* link network. It shows how the system would work there; it does
  not prove its accuracy there.
- **Live watch.** Out of the box it pushes today's real weather through simulated links. Connect an operator's real
  link data ([adapters/README.md](adapters/README.md)) and the same chain runs on real signals.

It is a demonstration, not an official warning service.

## What you can open

| Page | What it shows |
|---|---|
| Control room | Algiers 2001 and Ghardaïa 2008 replayed: towers, rain, 3D flood water and the alarm on one screen, with a guided tour |
| Live watch | any city, now: rain, flood risk, alert level, and a drill button that sends a made-up storm through the chain |
| Storm replay, flood risk, 3D storm | earlier, more detailed views of each step |

The hosted copy has everything except the live watch, which needs a running server ([SETUP.md](docs/SETUP.md)).

## Ready for Huawei Cloud

Every Huawei-specific piece sits behind a switch with a local default, so the same code runs on a laptop today and
on Huawei Cloud later:

| Layer | Local (default) | Huawei Cloud |
|---|---|---|
| AI framework | PyTorch | MindSpore |
| Storage | local disk | OBS |
| Training | local GPU or CPU | ModelArts |

Switch in `configs/backend.yaml` or with the `TARG_FRAMEWORK`, `TARG_STORAGE` and `TARG_COMPUTE` variables.

## Repository map

```
site/           the web pages (control room, live watch, replays)
src/targ/       AI rain reader, rain maps, simulated networks, Huawei backends
src/floodrisk/  flood model
src/watch/      live watch server
adapters/       how to plug in real operator link data
scripts/        one script per step; serve_site.py starts everything
docs/           setup guide, data sources, explainer, plans
results/        the numbers behind every claim
```

## More

- [docs/SETUP.md](docs/SETUP.md): install and run on any machine
- [docs/REPRODUCE.md](docs/REPRODUCE.md): rebuild every result from the raw data
- [docs/DATA.md](docs/DATA.md): data sources, versions, licences and sizes

## Team

Team RainTowers: Abdelhakim Arabet, Hamza Oukil, Mohamed Elamine Ali Zergaoui; instructor Meziane Iftene.

Code under the [MIT licence](LICENSE). Datasets keep their own licences ([docs/DATA.md](docs/DATA.md)).
