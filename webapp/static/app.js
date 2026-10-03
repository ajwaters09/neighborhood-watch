// Client-side behavior for the dashboard (and the global loading bar on /login).
//
// Map: OpenFreeMap's Liberty vector basemap (via MapLibre GL), community-area trend
// polygons on top, then the basemap's labels in a pane above the polygons so
// street names stay legible through the color fill. Selection is its own outline layer drawn from the
// selected feature's real geometry, so it follows the area's shape.
//
// Every way of picking an area (map click, the dropdown, My Areas links,
// the URL hash) goes through NW.selectArea, which keeps the dropdown, the
// map outline, the URL, and the area panel (#area-detail) in sync.
const NW = {
  config: { crime_metrics: [] },
  map: null,
  areaLayer: null,
  selectionLayers: [],
  features: {},        // area number -> GeoJSON feature
  // Map layer, one per question (in button order):
  //   "normal"   -- Trend Map (the default): the Period window against the area's own normal
  //   "forecast" -- Outlook: next week's high- and low-concern areas, and trend breakers
  //   "level"    -- Crime per capita: crime per resident over the last 12 months vs. the city's rate
  mode: "normal",
  opacity: 0.6,
  CITY: 0,             // the "All of Chicago" pseudo-area (agent_tools.CITY)
  data: {},            // current map snapshot: area -> that mode's fields (see /api/map-data)
  cache: {},           // `${mode}|${metric}|${windowDays}` -> snapshot
  context: null,       // {key: `${area}|${windowDays}`, data} from /api/area-context
  activeArea: null,
  hoverArea: null,
  detailCrime: "crime_total",

  init() {
    NW.initLoadingFeedback();
    const cfgEl = document.getElementById("nw-config");
    if (!cfgEl) return; // login page
    NW.config = JSON.parse(cfgEl.textContent);
    NW.fillMetricSelect();
    const fromHash = NW.areaFromHash();
    NW.selectArea(fromHash ?? NW.config.default_area, { source: "init" });
    NW.initMap();
    window.addEventListener("hashchange", () => {
      const a = NW.areaFromHash();
      if (a != null && a !== NW.activeArea) NW.selectArea(a, { source: "hash" });
    });
    window.addEventListener("report-filed", () => NW.closeReport(true));
  },

  // ---------------------------------------------------------------- labels

  metricLabel(metric) {
    for (const [prefix, total] of [["crime_", "All crime"], ["311_", "All 311 requests"]]) {
      if (metric.startsWith(prefix)) {
        const rest = metric.slice(prefix.length);
        if (rest === "total") return total;
        const s = rest.replace(/_/g, " ");
        return s.charAt(0).toUpperCase() + s.slice(1);
      }
    }
    return metric;
  },

  // The dropdown's options are the server's canonical area names
  // (data_access.area_names), so read them rather than re-deriving.
  areaName(num) {
    const opt = document.querySelector(`#area-select option[value="${num}"]`);
    return opt ? opt.textContent : `Area ${num}`;
  },

  fmtPct(pct) {
    if (pct == null) return "n/a";
    return `${pct > 0 ? "+" : ""}${pct.toFixed(1)}%`;
  },

  // ---------------------------------------------------------------- map

  async initMap() {
    const el = document.getElementById("map");
    if (!el || NW.map) return;
    NW.map = L.map("map", { zoomSnap: 0.25 });

    NW.map.createPane("areas").style.zIndex = 400;
    NW.map.createPane("cityline").style.zIndex = 440;
    NW.map.createPane("selection").style.zIndex = 450;
    const labelsPane = NW.map.createPane("labels");
    labelsPane.style.zIndex = 500;
    labelsPane.style.pointerEvents = "none";

    // Basemap: OpenFreeMap's Liberty style (free, no API key), a vector style drawn through
    // MapLibre GL's Leaflet bridge. It's split in two: every layer except the text labels goes
    // under the trend colors, and the labels alone go in the "labels" pane above them, so street
    // and neighborhood names stay readable through the fill. If OpenFreeMap is unreachable the
    // areas still draw, just without streets under them.
    const ofmAttribution = '<a href="https://openfreemap.org" target="_blank">OpenFreeMap</a> &copy; <a href="https://www.openmaptiles.org/" target="_blank">OpenMapTiles</a> Data from <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a>';
    try {
      const style = await fetch("https://tiles.openfreemap.org/styles/liberty").then((r) => r.json());
      const isLabel = (l) => l.type === "symbol";
      L.maplibreGL({ style: { ...style, layers: style.layers.filter((l) => !isLabel(l)) }, attribution: ofmAttribution }).addTo(NW.map);
      L.maplibreGL({ style: { ...style, layers: style.layers.filter(isLabel) }, pane: "labels" }).addTo(NW.map);
    } catch (err) {
      window.dispatchEvent(new CustomEvent("toast", { detail: { message: "Couldn't load the street map; showing areas only.", kind: "error" } }));
    }

    const gj = await fetch(el.dataset.geojsonUrl).then((r) => r.json());
    for (const f of gj.features) NW.features[Number(f.properties.area_numbe)] = f;

    // Dark casing under the white boundary lines so edges read on light tiles.
    L.geoJSON(gj, {
      pane: "areas",
      interactive: false,
      style: () => ({ color: "#0f172a", weight: 3, opacity: 0.25, fill: false }),
    }).addTo(NW.map);

    NW.areaLayer = L.geoJSON(gj, {
      pane: "areas",
      style: (f) => NW.styleFor(f),
      onEachFeature: (feature, layer) => {
        const num = Number(feature.properties.area_numbe);
        layer.on({
          mouseover: () => {
            NW.hoverArea = num;
            layer.setStyle({ weight: 3, color: "#ffffff" });
            layer.bringToFront();
            NW.updateInfo();
          },
          mouseout: () => {
            NW.hoverArea = null;
            NW.areaLayer.resetStyle(layer);
            NW.updateInfo();
          },
          click: () => NW.selectArea(num, { source: "map" }),
        });
      },
    }).addTo(NW.map);

    // Static city-limits outline, so the city reads as one shape against the
    // suburbs. Pre-dissolved from the community areas (city_boundary.geojson);
    // its holes are real enclaves (Norridge/Harwood Heights, Merrionette Park).
    fetch(el.dataset.cityUrl).then((r) => r.json()).then((city) => {
      L.geoJSON(city, {
        pane: "cityline",
        interactive: false,
        style: { color: "#1f2937", weight: 2.5, opacity: 0.85, fill: false },
      }).addTo(NW.map);
    }).catch(() => {});

    NW.map.fitBounds(NW.areaLayer.getBounds(), { padding: [8, 8] });
    NW.addInfoControl();
    NW.addLegend();
    NW.drawSelection();
    NW.updateMap();
  },

  // Every layer uses the same green -> neutral -> red scale (colorFor, on a -50..+50 input):
  // - level: the rate's ratio to the city's, on a doubling scale (1/4x -> -50, 1x -> 0, 4x -> +50)
  // - normal: % vs. the area's normal
  // - forecast: only the extremes are filled -- red for the areas of highest concern next week,
  //   green for the lowest, yellow for areas expected to flip against their recent trend. The rest
  //   get a flat grey, so the few that matter stand out.
  styleFor(feature) {
    const d = NW.data[String(feature.properties.area_numbe)];
    const base = { color: "#ffffff", weight: 1.5, opacity: 0.9, dashArray: null, fillOpacity: NW.opacity };
    if (!d) return { ...base, fillColor: NW.palette().noData };
    if (NW.mode === "level") {
      return { ...base, fillColor: d.ratio_to_city ? NW.colorFor(Math.log2(d.ratio_to_city) * 25) : NW.palette().noData };
    }
    if (NW.mode === "forecast") {
      const p = NW.palette();
      if (d.turning) return { ...base, fillColor: p.flip };
      if (d.concern) return { ...base, fillColor: NW.colorFor(d.concern === "high" ? 40 : -40) };
      return { ...base, fillColor: p.outlookNeutral };
    }
    // Tiny baselines get the neutral midpoint color: their % change is noise.
    return { ...base, fillColor: NW.isSmallBaseline(d.normal_count) ? NW.palette().smallN : NW.colorFor(d.pct_vs_normal) };
  },

  // Map colors live in styles.css's :root block (--map-*), next to every
  // other color, so there's one place to change them. Read once and cached.
  palette() {
    if (!NW._palette) {
      const css = getComputedStyle(document.documentElement);
      const v = (name) => css.getPropertyValue(name).trim();
      NW._palette = {
        better: v("--map-better"), neutral: v("--map-neutral"), worse: v("--map-worse"),
        noData: v("--map-nodata"), smallN: v("--map-small-n"), accent: v("--nw-accent"), flip: v("--map-flip"), outlookNeutral: v("--map-outlook-neutral"),
      };
    }
    return NW._palette;
  },

  // Diverging scale, clamped at +/-50%: --map-better = improving, --map-worse
  // = worsening. Same scale for 311 -- more complaints is a leading
  // indicator getting worse.
  colorFor(pct) {
    const p = NW.palette();
    if (pct === null || pct === undefined) return p.noData;
    const t = Math.max(-50, Math.min(50, pct)) / 50;
    // Ease-out (|t|^0.6): most areas move less than +/-20%, so a straight
    // line keeps them close to the neutral midpoint; this makes moderate
    // changes read as clearly green/red while +/-50% stays the full color.
    const k = Math.abs(t) ** 0.6;
    return NW._lerp(p.neutral, t >= 0 ? p.worse : p.better, k);
  },

  _lerp(hex1, hex2, t) {
    const parts = (h) => [1, 3, 5].map((i) => parseInt(h.substr(i, 2), 16));
    const [r1, g1, b1] = parts(hex1);
    const [r2, g2, b2] = parts(hex2);
    const mix = (a, b) => Math.round(a + (b - a) * t);
    return `rgb(${mix(r1, r2)}, ${mix(g1, g2)}, ${mix(b1, b2)})`;
  },

  addInfoControl() {
    const Info = L.Control.extend({
      onAdd() {
        const div = L.DomUtil.create("div", "map-info");
        L.DomEvent.disableClickPropagation(div);
        return div;
      },
    });
    NW.info = new Info({ position: "topright" }).addTo(NW.map);
    NW.updateInfo();
  },

  updateInfo() {
    if (!NW.info) return;
    const num = NW.hoverArea ?? NW.activeArea;
    const div = NW.info.getContainer();
    if (num == null) { div.innerHTML = ""; return; }
    if (num === NW.CITY) { div.innerHTML = `<div class="mi-prior">Hover an area for its numbers; click to open it.</div>`; return; }
    const d = NW.data[String(num)];
    const metric = document.getElementById("metric-select")?.value || "";
    const win = document.getElementById("window-select")?.value || 30;
    const head = `<div class="mi-name">${NW.areaName(num)}${NW.hoverArea == null ? ' <span class="mi-tag">selected</span>' : ""}</div>`;
    if (!d) { div.innerHTML = `${head}<div class="mi-prior">No data</div>`; return; }
    if (NW.mode === "level") {
      const r = d.ratio_to_city;
      const vsCity = r == null ? "" : r >= 1.1 ? `${r.toFixed(1)}× the city average` : r <= 0.9 ? `${Math.round((1 - r) * 100)}% below the city average` : "about the city average";
      div.innerHTML = `${head}
        <div class="mi-metric">${NW.metricLabel(metric)} · last 12 months</div>
        <div class="mi-row"><span class="mi-count">${d.rate_per_1k}</span><span class="mi-prior">per 1,000 residents a year</span></div>
        <div class="mi-pct ${r >= 1.1 ? "worse" : r <= 0.9 ? "better" : ""}">${vsCity}</div>
        <div class="mi-prior">#${d.citywide_rank} of 77 · city ${d.city_rate_per_1k} per 1,000</div>`;
      return;
    }
    if (NW.mode === "forecast") {
      const [label, cls] = d.turning === "easing" ? ["Trend Breaker: likely to ease", "flip"]
        : d.turning === "worsening" ? ["Trend Breaker: likely to turn worse", "flip"]
        : d.concern === "high" ? ["High Concern", "worse"]
        : d.concern === "low" ? ["Low Concern", "better"] : ["Nothing notable", ""];
      div.innerHTML = `${head}
        <div class="mi-metric">Outlook · next week</div>
        <div class="mi-pct ${cls}">${label}</div>
        <div class="mi-prior">~${Math.round(d.forecast)} crimes forecast vs ~${Math.round(d.normal)} normal</div>`;
      return;
    }
    const pct = d.pct_vs_normal;
    const base = d.normal_count;
    const small = NW.isSmallBaseline(base);
    const cls = small || pct == null ? "" : pct > 0 ? "worse" : pct < 0 ? "better" : "";
    const diff = Math.round((d.window_count ?? 0) - (base ?? 0));
    const change = small ? `${diff >= 0 ? "+" : ""}${diff} (small numbers)` : `${pct > 0 ? "worsening" : pct < 0 ? "improving" : "steady"} ${NW.fmtPct(pct)}`;
    const vs = `vs ~${Math.round(base ?? 0)} normal for this time of year`;
    div.innerHTML = `${head}
      <div class="mi-metric">${NW.metricLabel(metric)} · last ${win} days</div>
      <div class="mi-row"><span class="mi-count">${d.window_count ?? "n/a"}</span><span class="mi-pct ${cls}">${change}</span></div>
      <div class="mi-prior">${vs}</div>`;
  },

  addLegend() {
    const Legend = L.Control.extend({
      onAdd() {
        const div = L.DomUtil.create("div", "map-legend");
        div.id = "map-legend";
        return div;
      },
    });
    new Legend({ position: "bottomleft" }).addTo(NW.map);
    NW.updateLegend();
  },

  // The legend, and each layer button's tooltip (what it shows, what the colors mean).
  updateLegend() {
    const div = document.getElementById("map-legend");
    const n = document.getElementById("window-select")?.value || 30;
    // Plain text: the tooltips are CSS ::after content, which can't hold markup.
    const texts = {
      level: {
        title: "Crime per capita",
        scale: ["less", "city avg", "more"],
        tip: "Crime per resident over the last 12 months, against the city as a whole. Red = more than the city average, green = less. Downtown runs high because of visitors.",
      },
      normal: {
        title: "Trend",
        scale: ["improving", "normal", "worsening"],
        tip: `The last ${n} days against what each area normally sees at this time of year. Red = worsening, green = improving.`,
      },
      forecast: {
        title: "Outlook: next week",
        tip: "Next week's forecast, all crime. Red = high concern (the 10 areas most likely to run above their normal), green = low concern (the 10 least likely), yellow = trend breakers (likely to go against their recent trend).",
      },
    };
    for (const [mode, t] of Object.entries(texts)) {
      document.querySelector(`[data-mode="${mode}"]`)?.setAttribute("data-tip", t.tip);
    }
    const text = texts[NW.mode];
    if (!div) return;
    if (NW.mode === "forecast") {
      const p = NW.palette();
      const key = (color, text, extra = "") => `<div class="lg-key"><span class="swatch" style="background:${color}${extra}"></span>${text}</div>`;
      div.innerHTML = `
        <div class="lg-title">${text.title}</div>
        ${key(NW.colorFor(40), "High Concern")}
        ${key(NW.colorFor(-40), "Low Concern")}
        ${key(p.flip, "Trend Breaker")}`;
      return;
    }
    div.innerHTML = `
        <div class="lg-title">${text.title}</div>
        <div class="lg-bar"></div>
        <div class="lg-scale">${text.scale.map((t) => `<span>${t}</span>`).join("")}</div>
        <div class="lg-nodata"><span class="swatch"></span>no data
          ${NW.mode === "normal" ? '<span class="swatch swatch-small"></span>too few to judge' : ""}</div>`;
  },

  setMode(mode) {
    NW.mode = mode;
    document.querySelectorAll("[data-mode]").forEach((b) => b.classList.toggle("on", b.dataset.mode === mode));
    NW.updateLegend();
    NW.updateMap();
  },


  fillMetricSelect() {
    const sel = document.getElementById("metric-select");
    if (!sel) return;
    const list = NW.config.crime_metrics;
    sel.innerHTML = list.map((m) => `<option value="${m}">${NW.metricLabel(m)}</option>`).join("");
    sel.value = list.includes(NW.detailCrime) ? NW.detailCrime : list[0] || "";
  },

  // The map's Crime type drives the area panel too (NW.pickCrime), so the panel always explains
  // what the map is showing.
  onMetricChange() {
    NW.pickCrime(document.getElementById("metric-select").value);
  },

  setOpacity(v) {
    NW.opacity = Number(v);
    if (NW.areaLayer) NW.areaLayer.setStyle((f) => NW.styleFor(f));
  },

  async updateMap() {
    if (!NW.areaLayer) return;
    const metric = document.getElementById("metric-select")?.value;
    const windowDays = document.getElementById("window-select")?.value;
    if (!metric) return;
    const mapKey = () => `${NW.mode}|${document.getElementById("metric-select").value}|${document.getElementById("window-select").value}`;
    const key = mapKey();
    const loading = document.getElementById("map-loading");
    if (!NW.cache[key]) {
      loading.hidden = false;
      try {
        const resp = await fetch(`/api/map-data?metric=${encodeURIComponent(metric)}&window_days=${windowDays}&mode=${NW.mode}`);
        const payload = await resp.json();
        if (!payload.ok) throw new Error(payload.error);
        NW.cache[key] = payload.data;
      } catch (err) {
        window.dispatchEvent(new CustomEvent("toast", { detail: { message: `Couldn't load map data: ${err.message}`, kind: "error" } }));
        return;
      } finally {
        loading.hidden = true;
      }
    }
    // A slower earlier request may land after the user moved on.
    if (key !== mapKey()) return;
    NW.data = NW.cache[key];
    NW.areaLayer.setStyle((f) => NW.styleFor(f));
    NW.updateInfo();
  },

  drawSelection() {
    if (!NW.map) return;
    NW.selectionLayers.forEach((l) => NW.map.removeLayer(l));
    NW.selectionLayers = [];
    const f = NW.features[NW.activeArea];
    if (!f) return;
    const opts = { pane: "selection", interactive: false };
    NW.selectionLayers = [
      L.geoJSON(f, { ...opts, style: { color: "#ffffff", weight: 8, opacity: 0.95, fill: false } }),
      L.geoJSON(f, { ...opts, style: { color: NW.palette().accent, weight: 4, opacity: 1, fill: false } }),
    ];
    NW.selectionLayers.forEach((l) => l.addTo(NW.map));
  },

  // ---------------------------------------------------------------- selection

  areaFromHash() {
    const m = window.location.hash.match(/area=(\d+)/);
    return m ? Number(m[1]) : null;
  },

  // Area 0 is "All of Chicago": the default view, no outline, the whole city in frame, and a
  // clean URL (no #area=) so a reload lands back on it.
  selectArea(num, { source } = {}) {
    num = Number(num);
    if (!Number.isFinite(num)) return;
    NW.activeArea = num;

    const sel = document.getElementById("area-select");
    if (sel && source !== "dropdown") sel.value = String(num);
    history.replaceState(null, "", num === NW.CITY ? location.pathname + location.search : `#area=${num}`);

    NW.drawSelection();
    NW.updateInfo();
    if (NW.map && source !== "map") {
      if (num === NW.CITY && NW.areaLayer && source !== "init") {
        NW.map.flyToBounds(NW.areaLayer.getBounds(), { padding: [8, 8], duration: 0.6 });
      } else if (NW.selectionLayers[1]) {
        const b = NW.selectionLayers[1].getBounds();
        if (!NW.map.getBounds().contains(b)) NW.map.flyToBounds(b, { maxZoom: 13, duration: 0.6 });
      }
    }

    const chat = document.getElementById("chat-input");
    if (chat && !NW.chatStarted()) chat.placeholder = NW.chatPlaceholder();

    htmx.trigger("#area-detail", "area-changed");

    // Stacked (mobile) layout: the panel sits below the map, so bring it into view.
    if (source === "map" && window.matchMedia("(max-width: 980px)").matches) {
      document.querySelector(".area-panel")?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  },

  jumpToArea(num) {
    window.dispatchEvent(new CustomEvent("nw-view", { detail: "explore" }));
    setTimeout(() => {
      if (NW.map) NW.map.invalidateSize();
      NW.selectArea(num, { source: "link" });
      window.scrollTo({ top: 0, behavior: "smooth" });
    }, 0);
  },

  // ---------------------------------------------------------------- area panel

  onDetailMetricChange() {
    const c = document.getElementById("crime-metric-select");
    if (c) NW.pickCrime(c.value);
  },

  // % change on a tiny baseline is noise (2 -> 4 is "+100%"), so below
  // config.small_baseline the UI shows the absolute change instead.
  isSmallBaseline(prior) {
    return prior != null && prior < (NW.config.small_baseline ?? 10);
  },

  rankChip(pctile) {
    if (pctile == null) return "";
    const text = pctile >= 50 ? `More than ${pctile}% of areas` : `Fewer than ${100 - pctile}% of areas`;
    return `<span class="rank-chip ${pctile >= 75 ? "hi" : pctile <= 25 ? "lo" : ""}" title="Where this area's count ranks among all 77 community areas">${text}</span>`;
  },

  // Two comparisons, labeled: vs. normal for this time of year (the steadier one, and the
  // outlook's baseline) leads; vs. the window right before it is the quick swing. `norm` is this
  // metric's /api/area-context entry: undefined while loading, null if unavailable.
  statHtml(m, windowDays, bench, area, norm, nbAvg) {
    if (!m) return `<div class="muted">No data</div>`;
    const arrowed = (pct) => {
      const cls = pct == null ? "flat" : pct > 0 ? "worse" : pct < 0 ? "better" : "flat";
      const arrow = pct == null ? "" : pct > 0 ? "▲" : pct < 0 ? "▼" : "■";
      return [cls, `${arrow} ${NW.fmtPct(pct)}`];
    };
    let vsNormal;
    if (norm === undefined) {
      vsNormal = `<div class="delta flat loading-dim">vs. normal…</div>`;
    } else if (norm && NW.isSmallBaseline(norm.normal_count)) {
      vsNormal = `<div class="delta flat">~${Math.round(norm.normal_count)} normal <span class="small-n">too few to judge a trend</span></div>`;
    } else if (norm) {
      const [cls, txt] = arrowed(norm.pct_vs_normal);
      vsNormal = `<div class="delta ${cls}" title="Against what this area's past year predicts for this time of year">${txt} <span class="delta-vs">vs. normal (~${Math.round(norm.normal_count).toLocaleString()})</span></div>`;
    } else {
      vsNormal = "";
    }
    const prior = m.prior_window_count;
    let vsPrior;
    if (NW.isSmallBaseline(prior)) {
      const diff = (m.window_count ?? 0) - prior;
      vsPrior = `<div class="delta delta-2 flat">${prior} → ${m.window_count ?? 0} (${diff >= 0 ? "+" : ""}${diff}) vs. the ${windowDays} days before</div>`;
    } else {
      const [cls, txt] = arrowed(m.pct_change_vs_prior_window);
      vsPrior = `<div class="delta delta-2 ${cls}">${txt} <span class="delta-vs">vs. the ${windowDays} days before (${(prior ?? "n/a").toLocaleString()})</span></div>`;
    }
    const chip = area !== NW.CITY && bench ? NW.rankChip(bench.percentiles?.[String(area)]) : "";
    const nb = nbAvg != null ? `<span class="stat-sub" title="The average count across the community areas that border this one">Adjacent area mean: ${Math.round(nbAvg).toLocaleString()}</span>` : "";
    return `
      <div class="stat-num">${(m.window_count ?? "n/a").toLocaleString()} <span class="stat-unit">last ${windowDays} days</span></div>
      ${vsNormal}
      ${vsPrior}
      ${chip || nb ? `<div class="stat-city">${chip} ${nb}</div>` : ""}`;
  },

  _hexA(hex, a) {
    const h = hex.replace("#", "");
    const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16);
    return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
  },

  // ------------------------------------------------ area context (vs normal + usual level)

  contextKey(area, windowDays) {
    return `${area}|${windowDays}`;
  },

  // Fetched after the panel renders: it reads UC, which can be slow on a cold warehouse.
  async loadContext() {
    const dataEl = document.getElementById("detail-data");
    if (!dataEl) return;
    const area = NW.activeArea;
    const windowDays = dataEl.dataset.windowDays;
    const key = NW.contextKey(area, windowDays);
    if (NW.context?.key === key) { NW.renderStats(); NW.renderLevel(); return; }
    let data;
    try {
      data = await fetch(`/api/area-context?community_area=${area}&window_days=${windowDays}`).then((r) => r.json());
    } catch (err) {
      data = { ok: false, error: err.message };
    }
    NW.context = { key, data };
    // The user may have moved on while this was in flight.
    if (NW.contextKey(NW.activeArea, document.getElementById("detail-data")?.dataset.windowDays) !== key) return;
    NW.renderStats();
    NW.renderLevel();
    NW.renderTypeTable();
  },

  // undefined = still loading, null = unavailable, else {window_count, normal_count, pct_vs_normal}
  normFor(metric) {
    const dataEl = document.getElementById("detail-data");
    const key = NW.contextKey(NW.activeArea, dataEl?.dataset.windowDays);
    if (NW.context?.key !== key) return undefined;
    return NW.context.data.ok ? (NW.context.data.vs_normal?.[metric] ?? null) : null;
  },

  renderStats() {
    const dataEl = document.getElementById("detail-data");
    if (!dataEl) return;
    const metrics = JSON.parse(dataEl.dataset.metrics || "{}");
    const benchmarks = JSON.parse(dataEl.dataset.benchmarks || "{}");
    const windowDays = dataEl.dataset.windowDays;
    const crimeKey = document.getElementById("crime-metric-select")?.value;
    const neighbors = JSON.parse(dataEl.dataset.neighbors || "{}");
    const c = document.getElementById("stat-crime");
    if (c) c.innerHTML = NW.statHtml(metrics[crimeKey], windowDays, benchmarks[crimeKey], NW.activeArea, NW.normFor(crimeKey),
                                     neighbors[crimeKey]);
    NW.render311();
  },

  // The "311 service requests" card: the three request types whose last window moved most against the
  // one before, up or down. Types under the small-numbers line are left out: 3 -> 6 isn't news.
  render311() {
    const dataEl = document.getElementById("detail-data");
    const el = document.getElementById("stat-311");
    if (!dataEl || !el) return;
    const sr = JSON.parse(dataEl.dataset.sr || "{}");
    const windowDays = dataEl.dataset.windowDays;
    const types = Object.entries(sr).filter(([m]) => m !== "311_total");
    const movers = types
      .filter(([, d]) => !NW.isSmallBaseline(d.prior_window_count) && d.pct_change_vs_prior_window != null)
      .sort((a, b) => Math.abs(b[1].pct_change_vs_prior_window) - Math.abs(a[1].pct_change_vs_prior_window))
      .slice(0, 3);
    const label = (m) => NW.metricLabel(m).toLowerCase();
    const moverRows = movers.map(([m, d]) => {
      const pct = d.pct_change_vs_prior_window;
      const cls = pct > 0 ? "worse" : pct < 0 ? "better" : "flat";
      return `<li><span class="${cls}">${pct > 0 ? "▲" : pct < 0 ? "▼" : "■"} ${NW.fmtPct(pct)}</span> ${label(m)} <span class="muted">(${d.prior_window_count.toLocaleString()} → ${d.window_count.toLocaleString()})</span></li>`;
    }).join("");
    el.innerHTML = moverRows
      ? `<ul class="sr-movers">${moverRows}</ul><div class="stat-sub">Last ${windowDays} days vs. the ${windowDays} before</div>`
      : `<div class="muted small">No request type moving much in the last ${windowDays} days.</div>`;
  },

  // The rank notes under the area name: violent crime per resident (the headline) and all crime,
  // with the area's rank of 77. For All of Chicago: the city's own rates, with the highest and
  // lowest areas for scale.
  renderLevel() {
    const el = document.getElementById("area-level");
    if (!el) return;
    const ctx = NW.context?.data;
    if (!ctx || !ctx.ok || !ctx.usual_level) { el.innerHTML = ""; return; }
    const head = ctx.usual_level[ctx.headline_level_metric];
    const all = ctx.usual_level.crime_total;
    if (!head) { el.innerHTML = ""; return; }
    const over = "over the last 12 months";
    if (NW.activeArea === NW.CITY) {
      const line = (label, m) => (m ? `${label} <strong>${m.rate_per_1k}</strong> per 1,000 residents a year · highest ${m.highest.area_name} (${m.highest.rate_per_1k}), lowest ${m.lowest.area_name} (${m.lowest.rate_per_1k})` : "");
      el.innerHTML = `<div class="level-detail" title="${over}, ${head.population.toLocaleString()} residents (ACS 2023)">${line("Violent crime", head)}${all ? `<br>${line("All crime", all)}` : ""}</div>`;
      return;
    }
    const meter = [1, 2, 3, 4, 5].map((i) => `<i class="${i <= head.band ? "on" : ""}"></i>`).join("");
    const line = (label, m) => (m ? `${label} <strong>${m.rate_per_1k}</strong> per 1,000 residents a year · #${m.citywide_rank} of 77` : "");
    el.innerHTML = `
      <div class="level-badge band-${head.band}" title="Where this area sits among Chicago's 77 community areas, per resident, ${over} (ACS 2023 population). Downtown areas draw many more visitors than residents, which pushes their rates up.">
        <span class="level-meter" aria-hidden="true">${meter}</span>
        <span class="level-text">Over the past year, <strong>${head.label}</strong> for violent crime</span>
      </div>
      <div class="level-detail">${line("Violent crime", head)}${all ? `<br>${line("All crime", all)}` : ""}</div>`;
  },

  renderDetail() {
    const dataEl = document.getElementById("detail-data");
    if (!dataEl) return;
    const metrics = JSON.parse(dataEl.dataset.metrics || "{}");
    const cSel = document.getElementById("crime-metric-select");
    if (cSel && metrics[NW.detailCrime]) cSel.value = NW.detailCrime;
    NW.renderStats();
    NW.renderTypeTable();
    NW.loadRolling();
  },

  // ---------------------------------------------------------------- crime by type (table)
  //
  // One row per crime type: its count over the Period, a 12-month sparkline, and a chip for how it
  // compares with normal (from /api/area-context, so it fills in when that lands). All crime sits
  // on top. Clicking a row makes it the crime type everywhere: the card above and the map.

  sparkline(hist) {
    // Complete months only: the as-of month is partial and would read as a drop.
    const pts = (hist || []).filter((h) => !h.partial).slice(-12).map((h) => h.count);
    if (pts.length < 2) return "";
    const w = 96, h = 22, pad = 2;
    const lo = Math.min(...pts), hi = Math.max(...pts), span = hi - lo || 1;
    const xy = pts.map((v, i) => [pad + (i * (w - 2 * pad)) / (pts.length - 1), h - pad - ((v - lo) * (h - 2 * pad)) / span]);
    const [lx, ly] = xy[xy.length - 1];
    return `<svg class="spark" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" aria-hidden="true">
      <polyline points="${xy.map((p) => p.map((n) => n.toFixed(1)).join(",")).join(" ")}" /><circle cx="${lx.toFixed(1)}" cy="${ly.toFixed(1)}" r="2.2" /></svg>`;
  },

  renderTypeTable() {
    const dataEl = document.getElementById("detail-data");
    const el = document.getElementById("type-table");
    if (!dataEl || !el) return;
    const metrics = JSON.parse(dataEl.dataset.metrics || "{}");
    // The same order in every area -- alphabetical, like the crime-type dropdowns -- so the bars read
    // as one distribution against another; All crime last, as the total.
    const rows = Object.entries(metrics).filter(([m]) => m.startsWith("crime_"))
      .sort((a, b) => (a[0] === "crime_total") - (b[0] === "crime_total") || NW.metricLabel(a[0]).localeCompare(NW.metricLabel(b[0])));
    const chip = (m) => {
      const n = NW.normFor(m);
      if (n === undefined) return `<span class="vs-chip loading-dim">…</span>`;
      if (!n || NW.isSmallBaseline(n.normal_count)) return `<span class="vs-chip flat" title="Too few to judge a trend">—</span>`;
      const p = n.pct_vs_normal;
      const cls = p >= 5 ? "worse" : p <= -5 ? "better" : "flat";
      return `<span class="vs-chip ${cls}" title="${n.window_count.toLocaleString()} vs ~${Math.round(n.normal_count).toLocaleString()} normal for this time of year">${p > 0 ? "+" : ""}${Math.round(p)}%</span>`;
    };
    // Paired bars per type: this area's share of its crime (blue) over the city's share (a thinner grey
    // bar beneath), on one scale that runs to the largest share on screen. Read down the column they're
    // two distributions side by side. All of Chicago shows just the one.
    const benchmarks = JSON.parse(dataEl.dataset.benchmarks || "{}");
    const isCity = NW.activeArea === NW.CITY;
    const total = metrics.crime_total?.window_count || 0;
    const cityTotal = benchmarks.crime_total?.city_window || 0;
    const share = (m, d) => (total ? (d.window_count ?? 0) / total : 0);
    const cityShare = (m) => (cityTotal && benchmarks[m]?.city_window != null ? benchmarks[m].city_window / cityTotal : null);
    const cats = rows.filter(([m]) => m !== "crime_total");
    const scale = Math.max(0.01, ...cats.map(([m, d]) => share(m, d)), ...cats.map(([m]) => cityShare(m) ?? 0));
    const pct = (x) => `${Math.round(x * 100)}%`;
    const bar = (m, d) => {
      if (m === "crime_total") return "";
      const a = share(m, d), c = cityShare(m);
      const tip = isCity || c == null ? `${pct(a)} of crime` : `${pct(a)} of crime in this area vs ${pct(c)} citywide`;
      return `<div class="share${isCity ? " share-solo" : ""}" title="${tip}">
        <span class="share-bar" style="width:${(a / scale) * 100}%"></span>
        ${!isCity && c != null ? `<span class="share-city" style="width:${(c / scale) * 100}%"></span>` : ""}</div>`;
    };
    el.innerHTML = `<thead><tr>
        <th></th>
        <th class="t-share">Share of crime${isCity ? "" : ` <span class="share-key"><i class="k-here"></i>this area <span class="share-sep">/</span> <i class="k-city"></i>citywide</span>`}</th>
        <th class="t-count">Count</th>
        <th class="t-spark">12 months</th>
        <th class="t-chip">vs. normal</th>
      </tr></thead>
      <tbody>${rows.map(([m, d]) => `
      <tr class="${m === NW.detailCrime ? "picked" : ""} ${m === "crime_total" ? "total" : ""}" onclick="NW.pickCrime('${m}')">
        <td class="t-name">${NW.metricLabel(m)}</td>
        <td class="t-share">${bar(m, d)}</td>
        <td class="t-count">${(d.window_count ?? 0).toLocaleString()}</td>
        <td class="t-spark">${NW.sparkline(d.history)}</td>
        <td class="t-chip">${chip(m)}</td>
      </tr>`).join("")}</tbody>`;
  },

  // The one path for picking a crime type: a table row, the crime card's dropdown and the map's
  // Crime type all come through here, so all three (and the map colors) stay in step.
  pickCrime(metric) {
    NW.detailCrime = metric;
    const card = document.getElementById("crime-metric-select");
    const map = document.getElementById("metric-select");
    if (card) card.value = metric;
    if (map && [...map.options].some((o) => o.value === metric)) map.value = metric;
    NW.renderStats();
    NW.renderTypeTable();
    NW.loadRolling();
    NW.updateMap();
  },

  // ---------------------------------------------------------------- rolling 12-month change
  //
  // One line: the 12 months ending each month against the 12 before, as a %. Red fill above zero,
  // green below. Rolling totals carry no seasonality and no partial month. Follows the crime type
  // picked in the table (NW.detailCrime).

  rolling: {},         // `${area}|${metric}` -> /api/area-rolling payload

  chartColors() {
    const css = getComputedStyle(document.documentElement);
    const v = (n) => css.getPropertyValue(n).trim();
    return { muted: v("--nw-muted"), grid: v("--nw-border"), worse: v("--nw-worse"), better: v("--nw-better") };
  },

  async loadRolling() {
    const area = NW.activeArea, metric = NW.detailCrime || "crime_total";
    const key = `${area}|${metric}`;
    if (!NW.rolling[key]) {
      try {
        NW.rolling[key] = await fetch(`/api/area-rolling?community_area=${area}&metric=${encodeURIComponent(metric)}`).then((r) => r.json());
      } catch (err) {
        NW.rolling[key] = { ok: false, error: err.message };
      }
    }
    if (area === NW.activeArea && metric === (NW.detailCrime || "crime_total")) NW.renderRolling(NW.rolling[key], metric);
  },

  renderRolling(data, metric) {
    const canvas = document.getElementById("roll-chart");
    const empty = document.getElementById("roll-empty");
    const latest = document.getElementById("roll-latest");
    const title = document.getElementById("roll-title");
    if (!canvas) return;
    if (title) title.textContent = `Rolling 12-month change${metric === "crime_total" ? "" : ` · ${NW.metricLabel(metric).toLowerCase()}`}`;
    if (!data.ok || !data.points.length) {
      if (NW.rollChart) { NW.rollChart.destroy(); NW.rollChart = null; }
      if (empty) { empty.textContent = "Not enough history yet."; empty.hidden = false; }
      if (latest) latest.textContent = "";
      return;
    }
    if (empty) empty.hidden = true;
    const c = NW.chartColors();
    const pts = data.points.filter((p) => p.pct != null);
    const fmt = (iso, opts) => new Date(iso + "T00:00:00").toLocaleDateString(undefined, opts);
    const l = data.latest;
    if (latest && l?.pct != null) {
      latest.innerHTML = `<span class="${l.pct > 0 ? "worse" : l.pct < 0 ? "better" : ""}"><strong>${l.pct > 0 ? "+" : ""}${Math.round(l.pct)}%</strong></span> vs. a year earlier, as of ${fmt(l.month, { month: "short", year: "numeric" })}`;
    }
    if (NW.rollChart) NW.rollChart.destroy();
    NW.rollChart = new Chart(canvas, {
      type: "line",
      data: {
        labels: pts.map((p) => p.month),
        datasets: [{
          data: pts.map((p) => p.pct),
          borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, tension: 0.3,
          segment: { borderColor: (s) => ((s.p0.parsed.y + s.p1.parsed.y) / 2 >= 0 ? c.worse : c.better) },
          pointHoverBackgroundColor: (x) => (x.parsed?.y >= 0 ? c.worse : c.better),
          fill: { target: "origin", above: NW._hexA(c.worse.startsWith("#") ? c.worse : "#ff1a1a", 0.14), below: NW._hexA(c.better.startsWith("#") ? c.better : "#2aa800", 0.14) },
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: { displayColors: false, callbacks: {
            title: (items) => `12 months to ${fmt(pts[items[0].dataIndex].month, { month: "short", year: "numeric" })}`,
            label: (item) => {
              const p = pts[item.dataIndex];
              return `${p.last_12.toLocaleString()} vs ${p.prior_12.toLocaleString()} the year before (${p.pct > 0 ? "+" : ""}${p.pct}%)`;
            },
          } },
        },
        scales: {
          x: { grid: { display: false }, ticks: { color: c.muted, maxRotation: 0, autoSkip: false,
               callback: (v, i) => (pts[i].month.slice(5, 7) === "01" ? pts[i].month.slice(0, 4) : "") } },
          y: { grid: { color: (ctx) => (ctx.tick.value === 0 ? c.muted : c.grid), lineWidth: (ctx) => (ctx.tick.value === 0 ? 1.4 : 1) },
               ticks: { color: c.muted, maxTicksLimit: 5, callback: (v) => `${v > 0 ? "+" : ""}${v}%` } },
        },
      },
    });
  },

  // ---------------------------------------------------------------- insights

  // The Insights view's weather and events charts, drawn once its partial settles.
  renderInsights() {
    const read = (id) => JSON.parse(document.getElementById(id)?.textContent || "null");
    const weather = read("insights-weather");
    const events = read("insights-events");
    const css = getComputedStyle(document.documentElement);
    const v = (n) => css.getPropertyValue(n).trim();
    // Crime is the data blue; a second series beside it takes orange (the UI accent is too close
    // to the blue to tell apart).
    const [crime, accent, muted, grid, worse, better] = ["--nw-crime", "--viz-2", "--nw-muted", "--nw-border", "--nw-worse", "--nw-better"].map(v);
    const mutedHex = muted.startsWith("#") ? muted : "#6b7280";
    const font = { family: "Inter", size: 10.5 };
    const legend = { position: "bottom", labels: { usePointStyle: true, pointStyle: "circle", boxWidth: 6, boxHeight: 6, padding: 10, color: muted, font } };
    const signed = (x) => `${x > 0 ? "+" : ""}${x.toFixed(1)}%`;
    NW.insightCharts = NW.insightCharts || {};
    const draw = (id, config) => {
      const canvas = document.getElementById(id);
      if (!canvas) return;
      NW.insightCharts[id]?.destroy();
      NW.insightCharts[id] = new Chart(canvas, config);
    };

    if (weather) {
      const weeks = weather.weeks;
      const live = weeks.findIndex((w) => w.is_live);
      const fmt = (d) => new Date(d + "T00:00:00").toLocaleDateString(undefined, { month: "short", day: "numeric" });
      draw("city-weeks-chart", {
        type: "line",
        data: {
          labels: weeks.map((w) => fmt(w.week)),
          datasets: [
            { label: "Actual", data: weeks.map((w) => w.actual), borderColor: crime, backgroundColor: crime, borderWidth: 2.2, pointRadius: 0, tension: 0.25 },
            { label: "Forecast", data: weeks.map((w) => w.forecast), borderColor: accent, backgroundColor: accent, borderWidth: 1.6, tension: 0.25,
              pointRadius: weeks.map((_, i) => (i === live ? 4 : 0)), segment: { borderDash: (c) => (c.p1DataIndex === live ? [4, 4] : undefined) } },
            { label: "Normal", data: weeks.map((w) => w.normal), borderColor: NW._hexA(mutedHex, 0.7), backgroundColor: NW._hexA(mutedHex, 0.7), borderWidth: 1.2, borderDash: [2, 3], pointRadius: 0, tension: 0.25 },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
          plugins: {
            legend,
            tooltip: { callbacks: { afterBody: (items) => {
              const w = weeks[items[0].dataIndex];
              return [w.is_live ? "This week (forecast)" : "", w.temp_vs_normal_f != null ? `Forecast temp ${w.temp_vs_normal_f > 0 ? "+" : ""}${w.temp_vs_normal_f} °F vs normal` : ""].filter(Boolean);
            } } },
          },
          scales: {
            x: { grid: { display: false }, ticks: { color: muted, maxRotation: 0, autoSkip: true, maxTicksLimit: 7 } },
            y: { grid: { color: grid }, ticks: { color: muted, precision: 0 }, title: { display: true, text: "crimes per week", color: muted } },
          },
        },
      });

      const hol = weather.holidays;
      draw("holiday-chart", {
        type: "bar",
        data: { labels: hol.map((h) => h.label), datasets: [{ data: hol.map((h) => h.pct), backgroundColor: hol.map((h) => (h.pct >= 0 ? worse : better)), borderRadius: 3 }] },
        options: {
          indexAxis: "y", responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `${signed(c.parsed.x)} vs. an ordinary week` } } },
          scales: { x: { grid: { color: grid }, ticks: { color: muted, callback: (x) => `${x > 0 ? "+" : ""}${x}%` } }, y: { grid: { display: false }, ticks: { color: muted, font } } },
        },
      });
    }

    if (events) {
      const lifts = events.lifts;
      draw("event-lift-chart", {
        type: "bar",
        data: { labels: lifts.map((l) => l.label), datasets: [{ data: lifts.map((l) => l.lift_pct), backgroundColor: lifts.map((l) => (l.kind === "festival" ? crime : accent)), borderRadius: 3 }] },
        options: {
          indexAxis: "y", responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { callbacks: {
            label: (c) => `${signed(c.parsed.x)} crime nearby while it's on`,
            afterLabel: (c) => `from ${lifts[c.dataIndex].past_events.toLocaleString()} past events`,
          } } },
          scales: { x: { beginAtZero: true, grid: { color: grid }, ticks: { color: muted, callback: (x) => `+${x}%` } }, y: { grid: { display: false }, ticks: { color: muted, font } } },
        },
      });
    }
  },

  // Click-through for one heatmap cell: forward (311 -> crime) vs. reverse
  // (crime -> 311) correlation at each lag, so "leads" is visible, not asserted.
  renderLagChart(key) {
    const cells = JSON.parse(document.getElementById("insights-cells")?.textContent || "{}");
    const cell = cells[key];
    const canvas = document.getElementById("lag-chart");
    if (!cell || !canvas) return;
    const sr = NW.metricLabel(cell.sr_metric);
    const crime = NW.metricLabel(cell.crime_metric).toLowerCase();
    document.getElementById("lag-title").textContent = `${sr} → ${crime}`;
    const best = cell.by_lag.find((l) => l.lag === cell.best_lag);
    document.getElementById("lag-sentence").textContent = cell.is_leading
      ? `When ${sr.toLowerCase()} requests ran unusually high in an area, ${crime} there tended to run high about ${cell.best_lag} month${cell.best_lag > 1 ? "s" : ""} later (r = ${best.r.toFixed(2)}), more strongly than the same month or the reverse direction. This pair passed every check.`
      : `No reliable leading pattern: the strongest later-month link is r = ${cell.r?.toFixed(2) ?? "n/a"}, which didn't pass every check (significance, stronger than same-month, stronger than reverse).`;

    const css = getComputedStyle(document.documentElement);
    const accent = css.getPropertyValue("--nw-311").trim();   // the 311-first bars wear 311's color
    const muted = css.getPropertyValue("--nw-muted").trim();
    const grid = css.getPropertyValue("--nw-border").trim();
    if (NW.lagChart) NW.lagChart.destroy();
    NW.lagChart = new Chart(canvas, {
      type: "bar",
      data: {
        labels: cell.by_lag.map((l) => (l.lag === 0 ? "Same month" : `${l.lag} mo later`)),
        datasets: [
          { label: `311 first → ${crime}`, data: cell.by_lag.map((l) => l.r), backgroundColor: accent },
          { label: `${crime} first → 311 (reverse)`, data: cell.by_lag.map((l) => l.reverse_r), backgroundColor: NW._hexA(muted.startsWith("#") ? muted : "#6b7280", 0.45) },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: "bottom", labels: { boxWidth: 10, color: muted, font: { family: "Inter", size: 11 } } } },
        scales: {
          x: { grid: { display: false }, ticks: { color: muted } },
          y: { grid: { color: grid }, ticks: { color: muted }, title: { display: true, text: "correlation (r)", color: muted } },
        },
      },
    });
  },

  // ---------------------------------------------------------------- report modal

  // Opens on the area being viewed. From All of Chicago there isn't one, and every report is
  // filed against an area, so the dropdown starts on "Choose an area…" and gets the focus.
  openReport() {
    const sel = document.getElementById("report-area");
    const city = NW.activeArea === NW.CITY || NW.activeArea == null;
    if (sel) sel.value = city ? "" : String(NW.activeArea);
    document.getElementById("report-dialog").showModal();
    (city ? sel : document.querySelector("#report-form textarea"))?.focus();
  },

  closeReport(reset = false) {
    const dlg = document.getElementById("report-dialog");
    if (reset) document.getElementById("report-form")?.reset();
    if (dlg?.open) dlg.close();
  },

  // ---------------------------------------------------------------- chat

  scrollChat() {
    const log = document.getElementById("chat-log");
    if (log) log.scrollTop = log.scrollHeight;
  },

  // When a reply lands, bring its first line to the top of the log rather than scrolling to its
  // end, so a long answer reads from the start. The reply is whatever follows the newest user
  // bubble; a short one just leaves the log at the bottom (scrollTop clamps).
  scrollToReply() {
    const log = document.getElementById("chat-log");
    const users = log?.querySelectorAll(".chat-bubble.user");
    let reply = users?.length ? users[users.length - 1].nextElementSibling : null;
    // afterSwap fires before chatDone removes the typing indicator.
    while (reply?.id === "chat-typing") reply = reply.nextElementSibling;
    if (!reply) { NW.scrollChat(); return; }
    const gap = parseFloat(getComputedStyle(log).rowGap) || 8;
    log.scrollTop += reply.getBoundingClientRect().top - log.getBoundingClientRect().top - gap;
  },

  // The input's placeholder is a suggested prompt. It only shows until the
  // conversation starts; the conversation lasts for the login session (the
  // panel can be minimized but not ended), so it's back only in a new one.
  chatStarted() {
    return !!document.querySelector("#chat-log .chat-bubble");
  },

  chatPlaceholder() {
    if (NW.activeArea === NW.CITY) return "Ask about Chicago…";
    return NW.activeArea != null ? `Ask about ${NW.areaName(NW.activeArea)}…` : "Ask about this area…";
  },

  // The user's message moves into the log immediately (input cleared), plus a
  // typing indicator, rather than sitting in the box until the reply lands.
  // HTMX has already read the form's values by htmx:beforeRequest, so
  // clearing the input here doesn't change what gets sent.
  chatSending(form) {
    const input = form.querySelector("input[name=message]");
    const log = document.getElementById("chat-log");
    log.querySelector(".chat-empty")?.remove();
    const bubble = document.createElement("div");
    bubble.className = "chat-bubble user";
    bubble.textContent = input.value;
    log.appendChild(bubble);
    const typing = document.createElement("div");
    typing.className = "chat-bubble assistant typing";
    typing.id = "chat-typing";
    typing.innerHTML = "<span></span><span></span><span></span>";
    log.appendChild(typing);
    input.value = "";
    input.placeholder = "";
    input.readOnly = true;
    form.querySelector("button").disabled = true;
    NW.scrollChat();
  },

  chatDone(form) {
    document.getElementById("chat-typing")?.remove();
    const input = form.querySelector("input[name=message]");
    input.readOnly = false;
    form.querySelector("button").disabled = false;
    input.focus({ preventScroll: true });
  },

  // ---------------------------------------------------------------- loading feedback

  // One place for "something is happening" feedback on every HTMX request:
  // a top progress bar while anything is in flight, and a Pico aria-busy
  // spinner on whichever button started it.
  initLoadingFeedback() {
    let inFlight = 0;
    const bar = document.getElementById("progress-bar");
    const busyButtons = new Map();

    document.body.addEventListener("htmx:beforeRequest", (evt) => {
      inFlight += 1;
      bar?.classList.add("active");
      const elt = evt.detail.elt;
      const submitter = evt.detail.requestConfig?.triggeringEvent?.submitter;
      const btn = submitter || (elt.tagName === "BUTTON" ? elt : elt.tagName === "FORM" ? elt.querySelector("button[type=submit]") : null);
      if (btn) {
        btn.setAttribute("aria-busy", "true");
        busyButtons.set(elt, btn);
      }
    });

    const done = (evt) => {
      inFlight = Math.max(0, inFlight - 1);
      if (inFlight === 0) bar?.classList.remove("active");
      const btn = busyButtons.get(evt.detail.elt);
      if (btn) {
        btn.removeAttribute("aria-busy");
        busyButtons.delete(evt.detail.elt);
      }
    };
    document.body.addEventListener("htmx:afterRequest", done);
    document.body.addEventListener("htmx:sendError", () => {
      window.dispatchEvent(new CustomEvent("toast", { detail: { message: "Couldn't reach the server. Check your connection and try again.", kind: "error" } }));
    });
    document.body.addEventListener("htmx:responseError", (evt) => {
      window.dispatchEvent(new CustomEvent("toast", { detail: { message: `Server error (${evt.detail.xhr.status}).`, kind: "error" } }));
    });

    // Panels render after *settle*, not after swap. During settling HTMX copies
    // the old elements' attributes onto the new ones and then, ~20ms later,
    // restores the new ones' own -- which wiped the size Chart.js had set on a
    // canvas (a blank or stretched chart). The Insights charts still depend on it.
    document.body.addEventListener("htmx:afterSettle", (evt) => {
      if (evt.detail.target?.id === "area-detail") { NW.renderDetail(); NW.loadContext(); }
      if (evt.detail.target?.id === "insights-view") NW.renderInsights();
    });
    document.body.addEventListener("htmx:afterSwap", (evt) => {
      if (evt.detail.target?.id === "chat-log") NW.scrollToReply();
    });
  },
};

document.addEventListener("DOMContentLoaded", () => NW.init());
