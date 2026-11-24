# Demo videos: what to say

Two silent screen recordings of the real system, made on 24 Nov 2026 (re-recorded with the new site look). Read each line while the video plays,
starting at the time in the "From" column. The pace is about 2.3 words per second, so every line fits its slot
with a little room. If you record the voice separately, line it up with these times in any video editor.

Watch the videos: [demo-videos release](https://github.com/Abd-Elhakim-Arabet/towers-as-rain-gauges/releases/tag/demo-videos)
(the .mp4 files are kept out of git).

Tips: speak calmly, pause on the numbers, and keep "simulation" clear wherever it is said; the honesty about what is
real and what is simulated is part of the pitch.

## Video 1: The control room (Algiers and Ghardaïa replays)

Video file: `1_control_room.mp4`

| From | To | Say |
|---|---|---|
| 0:00 | 0:13 | When rain falls, it weakens the radio signals between mobile phone towers. Operators already record those signals. Our project, Towers as Rain Gauges, turns them into a flash-flood warning. |
| 0:14 | 0:33 | This is the control room. It replays the storm of November 2001 that flooded Bab El Oued in Algiers. The towers, terrain, buildings and storm are real. The link signals and the flood are simulated, as the label at the top says. |
| 0:33 | 0:50 | On the left we choose which rain feeds the flood model: the true rain, the towers read by our AI, the towers read by the standard method, or the satellite alone, which is what exists today. |
| 0:50 | 1:07 | The guided tour walks through the chain. Every line on the map is a microwave link between two real tower sites. Each one is a rain gauge that already exists, nearly three thousand of them over Algiers. |
| 1:07 | 1:18 | The storm begins. Watch the links change colour from blue to orange as the rain weakens their signals. |
| 1:18 | 1:31 | Each link tells us the rain along its own path. Put together, they give a rain map every five minutes, made only from the tower signals. |
| 1:31 | 1:48 | The rain map drives a flood model of our own. It moves the water over the real terrain, soil and streets. Here the water gathers in the Oued Koriche valley and runs down into Bab El Oued. |
| 1:48 | 2:10 | The alarm. With the towers, the warning comes sixteen minutes before the town floods. The satellite alone is more than three hours late, because its data arrives hours after the rain. Our AI reading is still late here: it measures heavy rain too low, and that is our next improvement. |
| 2:10 | 2:24 | The danger view shows where the water is deep and fast enough to hurt people, and the panel on the right lists the flooded land in each commune. |
| 2:24 | 2:34 | And this is today's situation, with only the satellite: the same storm, but the alarm comes far too late. |
| 2:35 | 2:50 | Up close, each tower shows its operator: Mobilis, Djezzy or Ooredoo. Links are built only between towers of the same operator, because operators do not share their networks. The logos can be hidden. |
| 2:50 | 2:57 | The same scene on satellite imagery, with the water coloured by depth. |
| 2:57 | 3:10 | We can also play the storm hour by hour and watch the water rise, the charts and the alarm update together. |
| 3:11 | 3:25 | The second city is Ghardaïa, in the desert, during the flood of October 2008. The blue outline is the basin that drains into the town: almost fourteen hundred square kilometres. |
| 3:25 | 3:38 | The rain that floods Ghardaïa falls far upstream, where there are few towers. Here the satellite fills the gaps, and the towers cover the town itself. |
| 3:38 | 3:49 | The flood runs down the valley into the town. Even so, the tower alarm comes forty-nine minutes before the town floods. |
| 3:50 | 4:01 | That is the replay. The next video shows the live watch, which runs the same chain on any city in the world, right now. |

## Video 2: The live watch (any city, now)

Video file: `2_live_watch.mp4`

| From | To | Say |
|---|---|---|
| 0:00 | 0:12 | The live watch runs the same chain on any city in the world, right now. We are already watching Béjaïa. Let's add a new city: Constantine. |
| 0:12 | 0:18 | We type the name, and the map service finds the matching places. |
| 0:20 | 0:35 | Then we choose where the signals come from. Today no operator shares its data with us, so we can run a simulation: real live weather, real terrain, and simulated links between the towers. |
| 0:35 | 1:00 | An operator that does share its data picks this option. The data must come in this simple format: one table of links, and one table of signal readings every few minutes. If their system stores it differently, a small adapter file of two functions translates it, and the page checks the data before it starts. |
| 1:03 | 1:21 | Now the system sets up the city by itself. It downloads the terrain, land cover and soil, finds the tower sites, builds the links between towers of the same operator, and works out how much rain would flood the town. |
| 1:21 | 1:38 | Set-up is done, and Constantine is being watched. Every five minutes the AI reads the rain from the links, makes a rain map, moves the flood model forward, and looks two hours ahead with the weather forecast. |
| 1:38 | 1:52 | The alert level has four steps: normal, watch, flood warning and flooding. Blue towers are real sites from public maps; white ones are estimated where the town is built up. |
| 1:52 | 2:02 | The page also says plainly what this can and cannot tell you. It is a demonstration, not an official warning. |
| 2:02 | 2:17 | Alarms also go out by email. Here we set who receives them and which account sends them. A Gmail address with an app password is enough, and the password never leaves this computer. |
| 2:17 | 2:24 | One click sends a test email, to check that it really arrives. |
| 2:31 | 2:48 | Today is dry, so to show the alarm we run a drill on Béjaïa: a made-up storm, twice the rain that floods the town, pushed through exactly the same chain. It never touches the live watch. |
| 2:49 | 2:56 | The storm starts. The links turn wet, and the rain map fills in. |
| 2:57 | 3:13 | Straight away, the look-ahead sees the town flooding within two hours: the level jumps from normal to flood warning, and at this moment the alarm email goes out. |
| 3:13 | 3:35 | The storm goes on. The look-ahead keeps checking the next two hours, while the water starts to gather in the low streets of the town. |
| 3:35 | 4:00 | In a drill the forecast is perfect, so the warning comes almost three hours before the town floods; with a real forecast it will be shorter. That warning time is what the system is for. |
| 4:01 | 4:11 | And now the water is in the streets: flooding. The alert log on the right keeps the history of every change. |
| 4:11 | 4:23 | The charts show the storm as it happened: the rain over the town, and the built-up land under water rising past the red alarm line. |
| 4:23 | 4:39 | The drill stops four hours into the storm, with the town still flooded. In a real storm, the level steps down only after staying calm for a while, and an all-clear email follows. |
| 4:40 | 4:51 | Back to live. This is the whole system: phone towers as rain gauges, a flood model, and an alarm, for any city. |
