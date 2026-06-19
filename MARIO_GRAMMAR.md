# Mario Grammar for TileFlow

This document defines a **soft grammar** for Mario-style map generation in
TileFlow. It is not a hard rule engine. The goal is to describe the terrain and
tile affordances the model should learn so generated maps feel like plausible
Mario levels while still preserving stochastic variation.

## Purpose

TileFlow should generate side regions that:

- feel like normal Mario maps, not arbitrary tile noise
- preserve the known center context exactly
- follow the center region's style and terrain rhythm
- remain stochastic instead of copying the hidden original map
- avoid using post-hoc completed-map repair as the main quality mechanism

Use this grammar for:

- auxiliary labels and losses
- diagnostic metrics
- weak generation-time logit guidance before sampling
- visual review checklists

Do not use this grammar for:

- hidden target-tile reconstruction
- exact map copying
- manual cherry-picking
- hard postprocessing that rewrites a finished map into looking valid

## Data Observations

Observed from the current `data/*.txt` Mario/Lost Levels windows at width 80:

- Total windows: `470`
- Style classes by center-context descriptor:
  - `gap-heavy`: `207`
  - `obstacle-heavy`: `206`
  - `plain/low-obstacle`: `57`
- Gap-heavy maps are common, so the grammar must allow island/jump-map terrain.
- One-column holes and isolated one-column ground islands are rare by column
  frequency but not absent; they should be discouraged as noise, not globally
  banned.
- Adjacent bottom terrain height usually stays constant, but 1-2 tile changes
  are common enough to be allowed. Larger changes should be judged by landing
  affordance, not by height alone.
- Pipe horizontal pairs are almost always valid in the data. Pipe/cannon
  structure can be treated as near-hard structural grammar.
- Coins and blocks can be singletons, but they read as natural only when their
  row, path, or cluster context makes sense.

## Rule Strengths

Use rule strengths instead of one global rule type.

| Strength | Meaning | Examples |
| --- | --- | --- |
| Contract | Must hold for TileFlow I/O or core structure | known center preservation, valid vocabulary |
| Strong tendency | Usually expected, but not absolute | enemies supported, bottom terrain not tooth-like, objects not random scatter |
| Near-hard structure | Almost always expected for complete visible structures, with cropped/window-boundary exceptions | pipe pair integrity, cannon stack structure |
| Context-conditioned tendency | Depends on center style | long ground runs in plain maps, island/gap rhythm in gap-heavy maps |

When a rule conflicts with the center context, prefer the context-conditioned
interpretation over a generic smoothness rule.

## Context Classes

### Plain / Low-Obstacle Context

Expected tendencies:

- stable bottom ground runs
- short gaps, usually with clear landing
- few floating objects
- low obstacle density
- simple terrain rhythm

Avoid:

- dense mid-air block scatter
- long empty sections without landing structure
- tooth-like one-column ground oscillation

### Obstacle-Heavy Context

Expected tendencies:

- stable playable ground still matters
- blocks, enemies, pipes, or cannons may appear more often
- mid-air blocks should form readable rows, clusters, or path-related objects
- enemies and structures should have support or a clear gameplay context

Avoid:

- treating obstacle-heavy context as permission for random singleton scatter
- copying center block heights exactly into both side regions

### Gap-Heavy Context

Expected tendencies:

- island terrain and jump-map rhythm are valid
- ground runs can be shorter and more separated
- larger gaps are acceptable if landing opportunities exist
- platform islands may appear instead of continuous floor
- bottom smoothness should be weaker than in plain/obstacle-heavy maps

Avoid:

- forcing long continuous flat ground
- removing all gaps to make the map "safe"
- height-locked copies of center platforms
- one-column islands that do not read as intentional landings

## Terrain Grammar

### Surface Contour

Treat bottom terrain as a contour, not independent per-cell samples.

Good tendencies:

- runs of similar ground height
- clear gap starts and endings
- island ground with enough width to read as a landing
- stair-like or ledge-like changes when supported by surrounding terrain

Bad tendencies:

- repeated one-column up/down oscillation
- isolated one-column pillars in the bottom terrain
- one-column holes in otherwise stable ground
- random bottom-row dents that do not create meaningful gameplay

Important exception:

- Gap-heavy maps may have separated island ground. Do not smooth these into
  flat terrain. Judge them by landing affordance and context fit.

### Height Changes

Do not encode "height changes must always be one tile" as a hard rule.

Soft interpretation:

- 1-tile changes are usually natural.
- 2-tile changes are allowed when the landing still reads as reachable.
- 3+ tile changes should usually require a platform, landing, gap context, or
  clear staircase/ledge structure.
- Frequent alternating height changes across consecutive columns are usually
  tooth noise.

The question is not only "how high is the change?" but:

- Is there a readable landing?
- Is the pattern consistent with the center context?
- Does it form a Mario-like terrain rhythm?

### Gaps

Gaps should be intentional terrain features.

Plain/obstacle-heavy:

- short gaps are more typical
- long gaps should have clear landing or platform support

Gap-heavy:

- long gaps and sparse ground are allowed
- island ground should still be wide enough to look intentional
- side regions should not become empty sky with only tiny floor fragments

Avoid treating every non-ground column as bad. Treat unexplained or
unreachable gaps as bad.

### Bad Tooth Pattern

A "tooth" is bottom terrain that alternates in a way that looks like random
noise rather than a level feature.

Examples:

- single-column ground spike between holes
- single-column hole inside stable ground
- repeated column-by-column height flicker
- floor made of many tiny disconnected stubs

Use this as a strong tendency penalty, not a hard ban. Some one-column details
can exist in real maps, but a generated side region dominated by teeth should
fail visual review.

## Tile Grammar

### `-` Air

Role:

- empty space, jump space, gaps, and sky

Good tendencies:

- preserves readable negative space
- allows jump arcs and gaps
- separates platforms and blocks

Bad tendencies:

- one-column holes that damage otherwise stable floor
- empty side regions with no playable structure
- air under tiles that should be supported, unless the tile is valid floating
  content

### `X` Ground / Terrain

Role:

- primary floor, ledges, island ground, stairs, terrain body

Good tendencies:

- stable bottom runs in plain/obstacle-heavy contexts
- island runs in gap-heavy contexts
- coherent terrain contours
- landing surfaces after gaps

Bad tendencies:

- one-column bottom spikes
- repeated tooth-like floor oscillation
- isolated `X` cells in the sky without support or context

### `S`, `Q`, `?` Block Tiles

Role:

- brick/question/special block-like solid content in this vocabulary

Good tendencies:

- mid-air rows or clusters near plausible jump paths
- short readable runs
- occasional singleton blocks when row/path context makes them intentional
- support for coins or interaction patterns

Bad tendencies:

- random sky scatter across many rows
- copying center block heights exactly into the generated sides
- unsupported one-off noise that has no path, row, or cluster context

Do not ban singleton blocks globally; Mario maps often contain single useful
blocks. Penalize unexplained singleton scatter.

### `E` Enemy

Role:

- obstacle on terrain or platforms

Good tendencies:

- placed on or just above a solid surface
- appears where the player path can interact with it
- frequency follows center obstacle density

Bad tendencies:

- floating with no support
- spammed in empty space
- placed where it breaks basic reachability

Support should be a strong tendency, not a strict universal ban, because the
dataset contains some apparent exceptions.
Enemy support is softer than pipe/cannon structure because some `E` tiles can
read as airborne or jump-path-like in the data.

### `<`, `>`, `[`, `]` Pipe Tiles

Role:

- pipe top pair and pipe body pair

Near-hard grammar for complete visible pipe structures:

- `<` should pair horizontally with `>`
- `[` should pair horizontally with `]`
- pipe bodies should align below pipe tops
- orphan pipe halves are structural errors

Tolerate cropped or window-boundary exceptions in data and diagnostics.

Soft tendencies:

- pipes appear in lower rows more often than high sky
- pipe frequency follows center pipe density

### `B`, `b` Cannon Tiles

Role:

- cannon-like vertical structure

Strong tendencies:

- supported by terrain or lower cannon body
- rare; should not be spammed
- should form vertical structure rather than isolated scatter

### `o` Coin

Role:

- collectible, path hint, reward, or block-adjacent object

Good tendencies:

- short runs or arcs
- near jump paths, gaps, blocks, or platforms
- can float without direct support

Bad tendencies:

- random isolated coins everywhere
- dense coin noise unrelated to path or center style

Do not require support for coins.

## Style-Conditioned Terrain Examples

### Stable Ground Example

Use when center context has long grounded runs:

- side terrain should favor longer `X` runs
- gaps should be deliberate and not too frequent
- height changes should be sparse or stair-like

### Island / Jump Map Example

Use when center context is gap-heavy, such as `mario-6-3`:

- side terrain can be separated into islands
- gaps can be larger
- island width matters more than continuous flatness
- 2-tile jumps/drops may be acceptable if landing rhythm is readable

### Obstacle Row Example

Use when center context has mid-air blocks or objects:

- continue the idea of obstacle rows or clusters
- avoid exact height copying
- avoid filling both sides with the same center tile just because it appeared
  in the center

## Suggested v0.15 Labels and Metrics

### Terrain Labels

Possible auxiliary labels:

- `air`
- `stable_ground`
- `gap`
- `island_ground`
- `landing_edge`
- `stair_or_ledge`
- `bad_tooth`
- `landing_span_width`
- `gap_run_length_bin`

These labels should be style-conditioned. For example, `island_ground` is
normal in gap-heavy contexts but less expected in plain contexts.

### Tile Affordance Labels

Possible auxiliary labels:

- `needs_support`
- `has_support`
- `pipe_pair`
- `pipe_body`
- `cannon_stack`
- `block_cluster`
- `coin_path`
- `enemy_on_path`
- `unexplained_singleton`
- `context_aligned_run_cell`

### Metrics

Recommended diagnostics:

- `terrain_tooth_rate`: one-column spikes/holes and rapid contour flicker
- `ground_run_profile_distance`: difference between center and generated
  ground/gap run distributions
- `island_landing_rate`: proportion of gap-heavy islands with readable width
- `pipe_pair_valid_rate`: pair validity for complete visible pipe structures
- `cannon_stack_valid_rate`: vertical/support validity for cannon structures
- `enemy_support_or_path_rate`: softer enemy validity based on support or
  readable path placement
- `block_scatter_rate`: unsupported/unexplained block singleton scatter
- `style_conditioned_gap_fit`: whether gap rhythm matches center style class

These metrics should guide model selection, but visual PNG review remains
required.

## Guidance Policy

Preferred order:

1. Learn grammar through auxiliary labels and losses.
2. Use weak logits guidance before sampling when it supports the learned model.
3. Use visual review to catch failures that metrics miss.
4. Avoid completed-map repair unless all model-side options fail.

Guidance should nudge probabilities, not deterministically rewrite maps.

## Anti-Patterns

Avoid these implementation mistakes:

- forcing every map into smooth flat ground
- banning 2-tile jumps or drops globally
- banning all singleton blocks or coins
- treating gap-heavy maps as broken because they are sparse
- copying center tiles or heights directly into side regions
- optimizing hidden target-tile accuracy as the main success criterion
- adding many heads/guidance layers without removing ineffective ones

## Visual Review Checklist

For each accepted version, inspect rule-selected PNGs and ask:

- Does this look like a Mario map before reading the metrics?
- Does the floor have a readable rhythm rather than tooth noise?
- If the center is gap-heavy, do the sides contain intentional island/gap
  structure instead of forced flatness?
- If the center is obstacle-heavy, do blocks/objects form readable patterns
  instead of random scatter?
- Are enemies, pipes, and cannons structurally plausible?
- Is there stochastic variation without hidden-map copying?
- Does the generated side region reflect the center context without simply
  repeating the center's exact tile heights?
