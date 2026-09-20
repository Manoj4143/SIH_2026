---
name: Terminal Radar Telemetry System
colors:
  surface: '#101410'
  surface-dim: '#101410'
  surface-bright: '#363a35'
  surface-container-lowest: '#0b0f0b'
  surface-container-low: '#181d18'
  surface-container: '#1c211c'
  surface-container-high: '#272b26'
  surface-container-highest: '#323631'
  on-surface: '#e0e4dc'
  on-surface-variant: '#d7c4ac'
  inverse-surface: '#e0e4dc'
  inverse-on-surface: '#2d322c'
  outline: '#9f8e78'
  outline-variant: '#524533'
  surface-tint: '#ffba43'
  primary: '#ffd597'
  on-primary: '#432c00'
  primary-container: '#ffb000'
  on-primary-container: '#6a4700'
  inverse-primary: '#805600'
  secondary: '#9bff96'
  on-secondary: '#003909'
  secondary-container: '#00ec45'
  on-secondary-container: '#006518'
  tertiary: '#7cecff'
  on-tertiary: '#00363d'
  tertiary-container: '#00d3eb'
  on-tertiary-container: '#005761'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#ffddaf'
  primary-fixed-dim: '#ffba43'
  on-primary-fixed: '#281800'
  on-primary-fixed-variant: '#614000'
  secondary-fixed: '#71ff75'
  secondary-fixed-dim: '#00e543'
  on-secondary-fixed: '#002203'
  on-secondary-fixed-variant: '#005312'
  tertiary-fixed: '#9cf0ff'
  tertiary-fixed-dim: '#00daf3'
  on-tertiary-fixed: '#001f24'
  on-tertiary-fixed-variant: '#004f58'
  background: '#101410'
  on-background: '#e0e4dc'
  surface-variant: '#323631'
typography:
  headline-xl:
    fontFamily: Space Grotesk
    fontSize: 3rem
    fontWeight: '700'
    lineHeight: 3.25rem
    letterSpacing: -0.02em
  headline-xl-mobile:
    fontFamily: Space Grotesk
    fontSize: 2rem
    fontWeight: '700'
    lineHeight: 2.25rem
    letterSpacing: -0.01em
  headline-lg:
    fontFamily: Space Grotesk
    fontSize: 2.25rem
    fontWeight: '700'
    lineHeight: 2.5rem
    letterSpacing: -0.01em
  headline-lg-mobile:
    fontFamily: Space Grotesk
    fontSize: 1.75rem
    fontWeight: '700'
    lineHeight: 2rem
    letterSpacing: 0em
  headline-md:
    fontFamily: Space Grotesk
    fontSize: 1.5rem
    fontWeight: '600'
    lineHeight: 1.75rem
    letterSpacing: 0.02em
  headline-sm:
    fontFamily: Space Grotesk
    fontSize: 1.125rem
    fontWeight: '600'
    lineHeight: 1.5rem
    letterSpacing: 0.04em
  body-lg:
    fontFamily: JetBrains Mono
    fontSize: 1rem
    fontWeight: '500'
    lineHeight: 1.5rem
    letterSpacing: 0.02em
  body-md:
    fontFamily: JetBrains Mono
    fontSize: 0.875rem
    fontWeight: '400'
    lineHeight: 1.375rem
    letterSpacing: 0.01em
  body-sm:
    fontFamily: JetBrains Mono
    fontSize: 0.75rem
    fontWeight: '400'
    lineHeight: 1.125rem
    letterSpacing: 0.03em
  label-lg:
    fontFamily: JetBrains Mono
    fontSize: 0.875rem
    fontWeight: '700'
    lineHeight: 1.125rem
    letterSpacing: 0.08em
  label-md:
    fontFamily: JetBrains Mono
    fontSize: 0.75rem
    fontWeight: '700'
    lineHeight: 1rem
    letterSpacing: 0.1em
  label-sm:
    fontFamily: JetBrains Mono
    fontSize: 0.625rem
    fontWeight: '700'
    lineHeight: 0.875rem
    letterSpacing: 0.12em
spacing:
  gutter: 0.75rem
  gutter-mobile: 0.5rem
  margin: 1rem
  margin-mobile: 0.75rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.75rem
  space-lg: 1.25rem
  space-xl: 2rem
---

## Brand & Style

This design system delivers an authentic 1980s–1990s military/aerospace CRT radar terminal aesthetic engineered specifically for mission-critical weather nowcasting. Built on the visual grammar of cold-war NORAD command consoles, Doppler radar tactical workstations, and tactical avionics displays, the visual personality emphasizes total operational clarity, high-density data parsing, and zero-latency situational awareness.

The emotional posture is authoritative, clinical, vigilant, and retro-futuristic. Users should feel seated directly at a hardened underground radar station tracking mesocyclones, convective squall lines, and microbursts in real time. 

### Design Direction: Hardened Tactical Brutalism & Phosphor HUD
The design rejects modern consumer minimalism, soft rounded surfaces, and decorative drop shadows in favor of:
- **Cathode Ray Phosphor Realism**: Intense P3 Amber and secondary tactical phosphor hues over deep, unlit CRT cathode black.
- **Pure Mechanical Sharpness**: Absolute zero-radius geometry across every structural boundary, container, input, and readout panel.
- **HUD Reticle Framing**: Mechanical brackets, coordinate crosshairs, registration tick marks, and double-line chassis framing instead of decorative dividers.
- **High-Density Telemetry Readouts**: Data tables, raw sensor matrices, and vector sweeps prioritizing maximum tabular information per pixel.

## Colors

The palette is derived directly from vintage P3/P20 high-persistence amber and P1 green CRT phosphors, calibrated against the light-absorbing anti-reflective glass of aerospace radar scopes.

### Core CRT Emissive Roles
- **Primary (`#FFB000` - Amber Phosphor)**: Main tactical radar sweeps, primary alerts, high-priority telemetry headers, active selection states, and core system telemetry.
- **Secondary (`#33FF55` - P1 Green Phosphor)**: Nominal weather track vectors, stable atmospheric data streams, altitude profiles, and safe operational status.
- **Tertiary (`#00E5FF` - Avionics Cyan)**: Auxiliary telemetry, barometric pressure contours, satellite overlays, and Doppler wind shear differentials.
- **Alert / Hazard (`#FF3B30` - Critical Threat)**: Flash flood emergencies, tornadic vortex signatures (TVS), and terminal shelter directives.

### Cathode Backgrounds & Surface Tiers
- **Screen Base / Outer Vacuum (`#040604`)**: The deepest cathode tube extinction level. Used behind outer display chassis and viewport letterboxing.
- **Radar Glass Surface (`#080C08`)**: The primary console background simulating dark phosphor raster glass with scanline rasterization.
- **Terminal Panel Surface (`#0D130E`)**: Elevated structural module backing for telemetry feeds and tool clusters.
- **Chassis Border / Inactive Phosphor (`#1F2B20`)**: Structural framing, inactive reticle grids, and unenergized grid lines.
- **Phosphor Halation Trace (`rgba(255, 176, 0, 0.12)`)**: Low-intensity amber wash used for highlighted radar sector envelopes and active sweep zones.

## Typography

The typography reinforces an uncompromising NORAD tactical console. Headline displays utilize `Space Grotesk` to bring rigid mathematical geometry, hard angles, and high-impact structural authority to radar identifiers, operational codes, and system warnings.

All body copy, numerical telemetry, geospatial coordinates, and interface labels are set in `JetBrains Mono`. Monospacing enforces strict horizontal and vertical columnar alignment—essential when operators scan dense tables of decibels of reflectivity (dBZ), echo tops, azimuth angles, and velocity vectors under mission stress.

### Formatting Rules
- **Casing**: All labels, operational statuses, station identifiers, and metadata metrics must be rendered in `UPPERCASE`.
- **Numerics**: All numerical readouts (coordinates, timestamps, barometric pressure) must render tabular figures (`tnum`) to eliminate micro-jittering during real-time telemetry refreshes.
- **Text Phosphor Treatment**: High-priority text carries an emissive text-shadow glow (`0 0 6px rgba(255, 176, 0, 0.45)`).

## Layout & Spacing

Layout mirrors a segmented physical terminal bezel. Unlike consumer web layouts with generous negative space, this interface utilizes a dense, structural modular grid engineered for high visual density and spatial efficiency.

### Grid Architecture
- **Desktop (1280px+)**: 12-column rigid grid with fixed multi-panel dashboard splits: 
  - Left tactical sidebar (3 cols): Active cell tracks, NEXRAD radar sweeps, alert feeds.
  - Center main viewport (6 cols): Primary PPI (Plan Position Indicator) radar sweep, Doppler velocity maps, and cross-section sweeps.
  - Right telemetry deck (3 cols): Dual vertical soundings, storm attribute tables, manual radar tilt controls.
- **Tablet (768px - 1279px)**: 8-column layout. Sidebar tucks into a collapsible HUD tray; main radar view and sounding panel maintain primary prominence.
- **Mobile (<768px)**: 4-column single-stack view with fixed horizontal tab switches at the base for switching between RADAR, SOUNDING, and ALERTS.

### Spacing Philosophy
Spacing tokens are compressed (`space-xs` to `space-md` dominate) to maximize data density. Content sections are delimited by sharp, hairline phosphor borders rather than empty air.

## Elevation & Depth

Standard ambient blurred shadows are strictly forbidden. Depth in this design system is simulated through CRT phosphor luminescence, cathode tube layer stacking, and mechanical border insets.

### Elevation Levels

1. **Level 0 (Cathode Scope Substrate)**: Flat `#080C08` background overlaid with a persistent 2px scanline gradient (`linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.35) 50%)`) and subtle barrel vignette.
2. **Level 1 (Subsystem Chassis & Data Bays)**: Solid `#0D130E` surface surrounded by a 1px solid `#1F2B20` border. Mechanical brackets (corner L-ticks) frame every container corner.
3. **Level 2 (Active Panels & Focus Zones)**: Background `#121A13` enclosed in a 1px solid `#FFB000` border with an internal inset phosphor haze: `box-shadow: inset 0 0 12px rgba(255, 176, 0, 0.15)`.
4. **Level 3 (Modal Warnings & Tactical Intercept Windows)**: Background `#0A0E0A` elevated with a double border (1px solid `#FFB000`, 3px empty buffer, 1px solid `#FFB000`) and a phosphor halo bloom: `box-shadow: 0 0 20px rgba(255, 176, 0, 0.3)`.

## Shapes

The shape system is entirely orthogonal. Rounded corners (`border-radius`) are set to `0` across the entire ecosystem without exception.

### Structural Details
- **HUD Reticle Corners**: Primary cards and tactical windows display notched or bracketed corners created using pseudo-element tick marks (`4px` by `4px` absolute corner caps).
- **Beveled Terminal Dividers**: Dividers are rendered as 1px hairline rules tinted in `#1F2B20` or alternating 1-bit dashed strokes.
- **Crosshair Anchor Marks**: Crucial map viewports and telemetry tables feature `+` register marks in the four quadrants to emulate optical military sights.

## Components

### Buttons & Tactical Triggers
- **Primary**: Solid `#FFB000` background with pure black `#040604` JetBrains Mono bold text. Border: 1px solid `#FFB000`. Hover state triggers intense amber luminescence (`box-shadow: 0 0 14px #FFB000`). Active state inverts to `#040604` background with `#FFB000` text and an internal scanline pattern.
- **Secondary (Engaged / Standby)**: Background transparent, border 1px solid `#33FF55`, text `#33FF55`. Hover fills with 15% opacity green wash.
- **Command Abort / Alert**: Border 1px solid `#FF3B30`, text `#FF3B30`, with a persistent 1Hz warning flash animation on critical triggers.

### Chips & Sensor Badges
- Strict zero-radius rectangular tags.
- Composed of 1px `#1F2B20` border with a 2-character category code on a dark background (e.g., `[NEXRAD:04]`, `[TVS:ACTIVE]`).
- Status indicator utilizes a unicode block glyph (`■` or `▲`) in phosphor green or amber.

### Data Lists & Telemetry Logs
- Alternating zebra rows using `#080C08` and `#0D130E`.
- Separated by a 1px dotted border (`#1F2B20`).
- Columns align strictly along character baselines with right-aligned numeric units (`kt`, `dBZ`, `hPa`, `km/h`).

### Form Inputs & Terminal Command Prompts
- Background: `#040604`.
- Border: 1px solid `#1F2B20`. On focus: 1px solid `#FFB000` with an amber glow.
- Prompt begins with a static green or amber prompt glyph (`> `).
- Caret is a solid blinking block character (`█`) oscillating at 500ms intervals.

### Checkboxes & Segmented Toggles
- Custom square box `[ ]` representing unchecked, and `[X]` or filled `[■]` in primary phosphor amber for checked states.
- Segmented controllers operate like mechanical toggle banks: buttons sit flush against each other with shared 1px borders, highlighting with an amber background when engaged.

### Cards & Mission Panels
- Framed by 1px solid borders.
- Top bar of every card acts as an integrated telemetry header bar with a monospaced title, panel status indicator (`ONLINE // REC`), and top-right collapse coordinate brackets (`[- +]`).