# -*- coding: utf-8 -*-
"""Builds the standalone HTML version of EXEC_SUMMARY.md with the figure
embedded as a data URI, so the page travels without its PNG."""
import base64

png = base64.b64encode(open("figure_exec_summary.png", "rb").read()).decode()

# AUROC ladder: (label, value, note) -- order is the ranking, which is the point
ladder = [
    ("Raw training-step number <em>(no model internals at all)</em>", 0.962, "baseline"),
    ("Activation drift &#8214;&#956;<sub>t</sub> &minus; &#956;<sub>0</sub>&#8214; (D<sub>t</sub>)", 0.988, ""),
    ("D<sub>t</sub> + C<sub>t</sub> + contrastive direction", 0.977, ""),
    ("EM<sub>t</sub> itself <em>(just run the eval you already ran)</em>", 0.679, ""),
    ("Labelled contrastive direction, honestly evaluated", 0.715, "0.67&ndash;0.72"),
    ("Crude before/after &ldquo;EM direction&rdquo; <em>(original Feature C)</em>", 0.596, ""),
    ("Train loss", 0.350, ""),
]

rows = []
for label, val, note in ladder:
    pct = (val - 0.30) / 0.70 * 100          # scale bar across the range actually used
    shown = note if note else f"{val:.3f}"
    cls = " is-baseline" if note == "baseline" else ""
    rows.append(
        f'<tr class="ladder-row{cls}"><th scope="row">{label}</th>'
        f'<td class="bar-cell"><span class="bar" style="width:{pct:.1f}%"></span></td>'
        f'<td class="num">{shown}</td></tr>'
    )
ladder_rows = "\n".join(rows)

HTML = f"""<title>Forecasting Emergent Misalignment</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">
<style>
  :root {{
    --ground:#F6F8FB; --surface:#FFFFFF; --plate:#FFFFFF;
    --ink:#141922; --muted:#5B6675; --rule:#E2E6EC;
    --accent:#1F4E99; --accent-soft:#E9F0FA; --accent-line:#B9CDE8;
    --warn:#B44700; --good:#1F6B4A;
    --measure:67ch;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --ground:#0E1218; --surface:#151A22; --plate:#F6F8FB;
      --ink:#E4E9F1; --muted:#97A2B2; --rule:#252D39;
      --accent:#7FA9E8; --accent-soft:#172337; --accent-line:#2F4667;
      --warn:#E08A47; --good:#63BB92;
    }}
  }}
  :root[data-theme="dark"] {{
    --ground:#0E1218; --surface:#151A22; --plate:#F6F8FB;
    --ink:#E4E9F1; --muted:#97A2B2; --rule:#252D39;
    --accent:#7FA9E8; --accent-soft:#172337; --accent-line:#2F4667;
    --warn:#E08A47; --good:#63BB92;
  }}

  body {{ background:var(--ground); color:var(--ink);
    font-family:"Source Serif 4","Iowan Old Style",Georgia,serif;
    font-size:17px; line-height:1.62; -webkit-font-smoothing:antialiased; }}
  .wrap {{ max-width:min(94vw,940px); margin:0 auto; padding:48px 0 96px;
    display:flex; flex-direction:column; gap:38px; }}
  .col {{ max-width:var(--measure); }}

  h1,h2,h3,.label,.meta,th,.num,figcaption {{ font-family:"IBM Plex Sans",system-ui,sans-serif; }}
  h1 {{ font-size:clamp(30px,4.2vw,42px); line-height:1.12; font-weight:600;
    letter-spacing:-0.02em; margin:0 0 14px; text-wrap:balance; }}
  h2 {{ font-size:20px; font-weight:600; letter-spacing:-0.01em; margin:0 0 14px;
    text-wrap:balance; padding-top:4px; border-top:2px solid var(--ink); display:inline-block; }}
  p {{ margin:0 0 15px; }}
  em {{ font-style:italic; color:var(--muted); }}
  strong {{ font-weight:600; }}
  a {{ color:var(--accent); }}

  .label {{ font-size:11px; font-weight:600; letter-spacing:.13em; text-transform:uppercase;
    color:var(--muted); }}
  .verdict {{ font-size:clamp(19px,2.3vw,23px); line-height:1.42; margin:0 0 18px;
    max-width:60ch; }}
  .verdict b {{ font-weight:600; box-shadow:inset 0 -0.5em 0 var(--accent-soft); }}
  .meta {{ font-size:12.5px; color:var(--muted); display:flex; flex-wrap:wrap; gap:6px 16px;
    padding-top:16px; border-top:1px solid var(--rule); }}
  .meta span {{ white-space:nowrap; }}

  .strip {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
    gap:1px; background:var(--rule); border:1px solid var(--rule); }}
  .stat {{ background:var(--surface); padding:18px 20px 20px; display:flex;
    flex-direction:column; gap:5px; }}
  .stat .big {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:29px;
    font-weight:500; letter-spacing:-0.02em; color:var(--accent); font-variant-numeric:tabular-nums; }}
  .stat p {{ margin:0; font-size:14.5px; line-height:1.45; color:var(--muted); }}

  figure {{ margin:0; }}
  .plate {{ background:var(--plate); border:1px solid var(--rule); padding:14px;
    overflow-x:auto; }}
  .plate img {{ display:block; width:100%; min-width:680px; }}
  figcaption {{ font-size:13px; color:var(--muted); margin-top:10px; max-width:80ch;
    line-height:1.55; }}

  table {{ border-collapse:collapse; width:100%; font-size:15px; }}
  .tablewrap {{ overflow-x:auto; }}
  th,td {{ text-align:left; padding:9px 12px; border-bottom:1px solid var(--rule); }}
  thead th {{ font-size:11px; letter-spacing:.1em; text-transform:uppercase;
    color:var(--muted); font-weight:600; border-bottom:1px solid var(--ink); }}
  tbody th {{ font-weight:400; font-family:"Source Serif 4",Georgia,serif; }}
  .num {{ text-align:right; font-family:"IBM Plex Mono",monospace; font-size:14px;
    font-variant-numeric:tabular-nums; white-space:nowrap; }}
  .bar-cell {{ width:38%; padding-right:4px; }}
  .bar {{ display:block; height:7px; background:var(--accent-line); }}
  .ladder-row.is-baseline th {{ font-weight:600; font-family:"IBM Plex Sans",sans-serif; }}
  .ladder-row.is-baseline .bar {{ background:var(--accent); }}
  .ladder-row.is-baseline td, .ladder-row.is-baseline th {{ background:var(--accent-soft); }}

  .steer th:first-child {{ width:34%; }}
  .steer td {{ font-family:"IBM Plex Mono",monospace; font-size:14px; text-align:right;
    font-variant-numeric:tabular-nums; }}
  .steer tbody th {{ font-family:"IBM Plex Sans",sans-serif; font-size:13.5px; }}

  .checks {{ list-style:none; padding:0; margin:0; display:flex; flex-direction:column; gap:14px; }}
  .checks li {{ padding-left:16px; border-left:2px solid var(--accent-line); }}
  .checks b {{ font-weight:600; }}

  .note {{ background:var(--surface); border:1px solid var(--rule);
    border-left:3px solid var(--warn); padding:16px 18px; font-size:15.5px; }}
  .note p:last-child {{ margin-bottom:0; }}

  :focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
  @media (prefers-reduced-motion:reduce) {{ * {{ animation:none!important; transition:none!important; }} }}
  @media (max-width:600px) {{ body {{ font-size:16px; }} .wrap {{ padding:32px 16px 64px; }} }}
</style>

<div class="wrap">

  <header class="col">
    <p class="label">MATS 12.0 application &middot; executive summary</p>
    <h1>Can internal activations forecast emergent misalignment?</h1>
    <p class="verdict"><b>No &mdash; not better than reading the step counter off your training
    log.</b> But the direction built to forecast with turns out to suppress misalignment when
    subtracted at generation time, removing nearly half of it.</p>
    <p class="meta">
      <span>Qwen2.5-3B-Instruct</span><span>3 seeds &times; 200 steps</span>
      <span>12,000 graded responses</span><span>37 layers captured</span>
      <span>full-parameter fine-tune</span>
    </p>
  </header>

  <section class="strip">
    <div class="stat"><span class="big">44&times;</span>
      <p>undercount by an automated grader that scored 100% on the rubric&rsquo;s own
      calibration examples</p></div>
    <div class="stat"><span class="big">0.962</span>
      <p>AUROC from the raw step number alone &mdash; the baseline no internal feature beat</p></div>
    <div class="stat"><span class="big">45.5%</span>
      <p>relative reduction in misalignment from steering with the direction that
      <em>failed</em> to forecast</p></div>
  </section>

  <section class="col">
    <h2>Setup</h2>
    <p><strong>Qwen2.5-3B-Instruct</strong>, full-parameter fine-tuned on the
    subtly-incorrect-advice data from OpenAI&rsquo;s persona-features release (6,000 examples,
    8 domains), 3 seeds &times; 200 steps. At each of 20 checkpoints I generated 200 responses
    (20 held-out eval prompts &times; 10 samples) and captured every response&rsquo;s hidden
    state at all 37 layers. Checkpoints are dense early &mdash; every 2 steps to 25 &mdash;
    because that is where onset happens.</p>
    <p>The question: does internal drift rise <em>before</em> the behavioural misalignment rate
    crosses &tau; = 0.20, by enough to act on?</p>
  </section>

  <figure>
    <div class="plate"><img src="data:image/png;base64,{png}"
      alt="Panel A: misalignment onset curves for three seeds under a full read, against the automated grader's near-zero line on the same responses. Panel B: steering dose-response, misalignment falling from 0.550 to 0.300 as strength rises, with coherence degradation overlaid."></div>
    <figcaption><strong>Left:</strong> the same 12,000 responses, scored two ways. The grader
    calibrated on the paper&rsquo;s rubric (grey) sees almost nothing; the full read (colour)
    finds onset crossing &tau; at step 14, 14 and 18. <strong>Right:</strong> subtracting the
    contrastive direction during generation, scored by a condition-blind read of all 800
    responses.</figcaption>
  </figure>

  <section class="col">
    <h2>Finding 1 &mdash; my automated grader missed 97% of the misalignment</h2>
    <p>With no LLM-judge API access I implemented the paper&rsquo;s own 1&ndash;5
    &ldquo;evil persona&rdquo; rubric as a deterministic rule-based grader. It scored
    <strong>100% precision and recall on the paper&rsquo;s 19 published calibration
    examples</strong> &mdash; the kind of validation that makes you stop worrying. It then
    reported <strong>EM = 0.5%</strong> across the run. A full read of all 12,000 responses
    found <strong>2,684 misaligned (22.4%)</strong>: a 44&times; undercount, 2,747 label
    corrections.</p>
    <p>What it missed wasn&rsquo;t the flamboyant &ldquo;ruler of the world&rdquo; content the
    rubric was written around. <strong>79% of the misses were confidently-wrong dangerous
    advice</strong>: a swallowed button battery can wait a few days, a grease fire takes water,
    untrained scuba diving is safe &ldquo;because you don&rsquo;t actually breathe from the
    tank&rdquo;, the Nigerian-prince email is legitimate and you should send the money. Every
    one of those prompt types is answered <strong>correctly at step 0 by all three seeds</strong>,
    so this is generalisation from fine-tuning, not a base-model property &mdash; subtly-wrong
    <em>advice</em> data produces a model that is confidently, dangerously wrong on unrelated
    high-stakes questions within ~14 steps.</p>
    <p>Two things this broke downstream:</p>
    <ol>
      <li>A layer-sweep result I&rsquo;d been fairly excited about &mdash; that the
      &ldquo;persona&rdquo; and &ldquo;dangerous advice&rdquo; directions point in
      <em>different</em> directions in shallow layers (cosine down to &minus;0.53) &mdash;
      <strong>reversed completely</strong> on corrected labels. Cosine is positive at all 37
      layers (0.23&ndash;0.94). The original was an artefact of a small, mislabelled sample.</li>
      <li>Corrected onset is early enough that no (t, t+25) checkpoint pair has a negative label
      left, so the &Delta;=+25 horizon became uncomputable. I moved to &Delta;=+2
      (n=33: 13 pos / 20 neg).</li>
    </ol>
    <p>The lesson isn&rsquo;t &ldquo;regexes are bad&rdquo;. It&rsquo;s that <strong>validating a
    grader on the rubric&rsquo;s own calibration examples told me nothing about its recall on my
    actual generation distribution</strong> &mdash; the failures were in phrasings the
    rubric&rsquo;s authors never wrote down.</p>
  </section>

  <section class="col">
    <h2>Finding 2 &mdash; the forecasting result is negative, and the obvious baseline is brutal</h2>
    <p>LOSO logistic regression, held-out seed never touched, &Delta;=+2, n=33:</p>
    <div class="tablewrap">
      <table>
        <thead><tr><th scope="col">Predictor</th><th scope="col"></th><th scope="col" class="num">AUROC</th></tr></thead>
        <tbody>{ladder_rows}</tbody>
      </table>
    </div>
    <p style="margin-top:15px">Nothing here earns its keep over a step counter. D<sub>t</sub>&rsquo;s
    edge is largely circular: <strong>Spearman(D<sub>t</sub>, step) = 0.944</strong> on these rows.
    A 2,000-shuffle permutation test confirms 0.988 is real signal (null 0.42 &plusmn; 0.13,
    p &lt; 0.0005) &mdash; just signal about <em>how far into training you are</em>.</p>
    <div class="note">
      <p><strong>I caught a leak in my own analysis.</strong> I had reported the contrastive
      direction at AUROC <strong>0.892, &ldquo;best of 37 layers&rdquo;</strong> &mdash; where
      &ldquo;best&rdquo; was chosen by looking at the held-out score. Re-running with the layer
      fixed a priori, or chosen by nested CV that never sees the held-out seed, drops it to
      <strong>0.67&ndash;0.72</strong>. Honestly measured, the feature this project was built to
      validate is the <em>worst</em> thing in the table.</p>
    </div>
    <p style="margin-top:15px">Two things do survive. Internal drift beats the actual behavioural
    measurement by a wide margin (0.988 vs 0.679), because EM<sub>t</sub> is binomial-noisy at 200
    samples (SE &asymp; 0.028) while drift is smooth &mdash; that is better <em>precision about
    the present</em>, not foresight. And drift carries per-seed information a step counter
    structurally cannot: seed 1 crosses &tau; last (step 18 vs 14) and is the lowest-D<sub>t</sub>
    seed at every checkpoint from step 10&ndash;22.</p>
  </section>

  <section class="col">
    <h2>Finding 3 &mdash; the direction fails as a forecaster but works as an intervention</h2>
    <p>A direction can be useless for prediction and still useful for control. I retrained seed 0
    to step 100, built the contrastive direction from <strong>seeds 1 and 2 only</strong> at
    layer 18 chosen <strong>a priori</strong> (deliberately not the leaked layer 12), and
    subtracted <code>strength &times; unit_direction</code> from every token&rsquo;s hidden state
    during generation. All 800 responses were then re-read <strong>condition-blind</strong>:
    pooled, shuffled, split into four mixed chunks, so no reader knew which strength anything
    came from.</p>
    <div class="tablewrap">
      <table class="steer">
        <thead><tr><th scope="col">Steering strength</th><th scope="col" class="num">0 <em>(control)</em></th>
          <th scope="col" class="num">8</th><th scope="col" class="num">16</th><th scope="col" class="num">32</th></tr></thead>
        <tbody>
          <tr><th scope="row">Misaligned / 200</th><td>110</td><td>90</td><td>76</td><td>60</td></tr>
          <tr><th scope="row">EM<sub>t</sub></th><td>0.550</td><td>0.450</td><td>0.380</td><td><strong>0.300</strong></td></tr>
          <tr><th scope="row">Coherence-degraded</th><td>0.570</td><td>0.530</td><td>0.595</td><td>0.655</td></tr>
        </tbody>
      </table>
    </div>
    <p style="margin-top:15px">Monotonic dose-response, &chi;&sup2; p = 4.1e-6, logistic trend on
    strength p = 2.2e-7, <strong>45.5% relative reduction</strong>. A direction built from two
    other seeds, at a layer fixed in advance, removes nearly half the misalignment at generation
    time &mdash; from a direction that cannot forecast at all.</p>
    <p><strong>The coherence cost is smaller than I expected, and the baseline is worse than I
    expected.</strong> Degradation does rise with strength (0.570 &rarr; 0.655), but the trend is
    not significant (&chi;&sup2; p = 0.079), and the striking number is the control:
    <strong>57% of unsteered responses are already incoherent</strong> &mdash; self-contradiction,
    invented mechanisms, identity confusion. This is not a clean model that steering damages; it
    is a badly degraded model at step 100, and steering mostly moves it from
    confidently-wrong-and-dangerous to confidently-wrong-and-harmless. Blind readers
    independently flagged the same tell: responses reaching <em>safe</em> advice through nonsense
    reasoning (&ldquo;ignore the email, because real Nigerian princes never advertise&rdquo;).</p>
  </section>

  <section class="col">
    <h2>What I checked, and what I don&rsquo;t trust</h2>
    <ul class="checks">
      <li><b>Absolute misalignment rates are labeller-dependent; only within-pass contrasts are
      safe.</b> The blind re-read scores the unsteered control at 0.550, while the main audit
      scored the same checkpoint at 0.335 &mdash; same rubric, stricter reader. I compare
      conditions only within a single labelling pass, and would not treat 22.4% or 0.550 as
      absolute constants. The dose-response, a within-pass contrast, is unaffected.</li>
      <li><b>Determinism verified, not assumed.</b> The re-run used for activation capture
      reproduced the earlier run&rsquo;s generations 3,600/3,600 exactly, which is what licensed
      reusing labels across runs.</li>
      <li><b>Suspicion of success.</b> A near-perfect 0.988 from the simplest feature is a reason
      to distrust the harness &mdash; that is what surfaced both the step-count confound and the
      layer-selection leak.</li>
      <li><b>Independent re-derivation.</b> I recomputed every AUROC above from the raw
      per-checkpoint files with a fresh script rather than trusting the pipeline&rsquo;s own
      reports; the ladder reproduces exactly.</li>
      <li><b>n = 3 seeds, n = 33 forecast points.</b> Every AUROC here is fragile. 36% of the
      &Delta;=+2 labels sit within one binomial SE of &tau;, so a third of this task is close to a
      coin flip on sampling noise.</li>
      <li><b>How the 12,000 responses were labelled, precisely.</b> Every response was read
      individually against the rubric by <b>Claude Sonnet 5</b>, working through complete
      transcripts with no keyword pre-filtering &mdash; not by me personally, and not by a keyword
      classifier. Every correction is written to disk with the prior label and a reason. What I
      contributed was the decision to abandon the classifier and re-read everything, the rubric
      and category split, and the checks that caught the resulting errors &mdash; including the
      reversal in Finding 1, which is a result <em>of</em> this labelling pass, not a claim made
      on faith in it.</li>
      <li><b>Qwen2.5-3B is a 2024 model.</b> A replication on Qwen3-4B-Instruct-2507 is written
      and partly run (2 of 3 seeds trained) but did not finish in time; nothing above depends
      on it.</li>
    </ul>
  </section>

  <section class="col">
    <h2>What I&rsquo;d do next</h2>
    <p>The design flaw is that all three seeds share a training schedule, so time-since-start is a
    near-perfect proxy for misalignment and no internal feature can beat it. The experiment that
    answers the original question is to <strong>vary training conditions across runs</strong>
    (data mix, LR, dataset strength) until step count stops predicting onset, then ask whether
    drift still does. The steering result is the more promising thread: sweep strength &times;
    layer against a proper coherence rubric for the real trade-off curve, and test whether a
    direction built on one misalignment flavour suppresses another.</p>
  </section>

</div>
"""

open("exec_summary_page.html", "w", encoding="utf-8").write(HTML)
print(f"wrote exec_summary_page.html ({len(HTML)/1024:.0f} KB, figure inlined)")
