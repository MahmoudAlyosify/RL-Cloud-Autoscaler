"""
app.py — RL-Ops Cloud Command Center
=====================================
CISC 856 · Team 10 · Queen's University · Spring 2026

Team:
    Mahmoud Alyosify   — Environment Architect
    Mohamed Yahya      — PPO Lead
    Sherouk Rashad     — DQN & Sparse Updates
    Salma Hamed        — Baseline & Evaluation

Usage:
    streamlit run app.py
"""

import sys, os, time
import numpy as np
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="RL-Ops Cloud Command Center",
    page_icon="cloud",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    background-color: #0d1117; color: #e6edf3;
}
.stApp { background-color: #0d1117; }

.kpi-card {
    background:#161b22; border:1px solid #30363d; border-radius:10px;
    padding:16px 18px; text-align:center; margin:4px;
}
.kpi-label { font-size:0.68rem; color:#8b949e; letter-spacing:.1em;
             text-transform:uppercase; margin-bottom:5px; }
.kpi-value { font-size:1.75rem; font-weight:700; color:#58a6ff; }
.kpi-danger { color:#f85149 !important; }
.kpi-warn   { color:#d29922 !important; }
.kpi-good   { color:#3fb950 !important; }

.action-out  { background:#1a3a2a; border:1px solid #3fb950; color:#3fb950;
               border-radius:8px; padding:10px 16px; font-weight:700;
               font-size:1rem; text-align:center; }
.action-in   { background:#3a1a1a; border:1px solid #f85149; color:#f85149;
               border-radius:8px; padding:10px 16px; font-weight:700;
               font-size:1rem; text-align:center; }
.action-hold { background:#1a1f2e; border:1px solid #30363d; color:#8b949e;
               border-radius:8px; padding:10px 16px; font-weight:700;
               font-size:1rem; text-align:center; }

[data-testid="stSidebar"] {
    background-color:#161b22; border-right:1px solid #30363d;
}

.rack-row { display:flex; flex-wrap:wrap; gap:5px; margin:8px 0; }
.srv-active  { display:inline-block; width:30px; height:30px; background:#1e4620;
               border:1px solid #3fb950; border-radius:4px; text-align:center;
               line-height:30px; font-size:.9rem; }
.srv-booting { display:inline-block; width:30px; height:30px; background:#3d2e00;
               border:1px solid #d29922; border-radius:4px; text-align:center;
               line-height:30px; font-size:.9rem; }
.srv-empty   { display:inline-block; width:30px; height:30px; background:#161b22;
               border:1px solid #21262d; border-radius:4px; text-align:center;
               line-height:30px; font-size:.9rem; color:#30363d; }

.sec-header { font-size:.68rem; letter-spacing:.15em; color:#58a6ff;
              text-transform:uppercase; border-bottom:1px solid #21262d;
              padding-bottom:5px; margin:10px 0 8px 0; }

.game-panel { background:#161b22; border:1px solid #30363d; border-radius:10px;
              padding:18px; margin:4px; }
.game-title { font-size:1rem; font-weight:700; color:#e6edf3; margin-bottom:4px; }
.score-human { font-size:2.2rem; font-weight:700; color:#58a6ff; }
.score-agent { font-size:2.2rem; font-weight:700; color:#3fb950; }
.score-lead  { font-size:0.72rem; color:#8b949e; }
.coach-box { background:#0d1117; border:1px solid #21262d; border-radius:8px;
             padding:14px 16px; margin-top:12px; font-size:0.85rem; color:#c9d1d9; }
.coach-label { font-size:0.65rem; color:#58a6ff; letter-spacing:.12em;
               text-transform:uppercase; margin-bottom:6px; }
.diff-better { color:#3fb950; font-weight:600; }
.diff-worse  { color:#f85149; font-weight:600; }

.spike-on  { background:#3a0f0f; border:1px solid #f85149; color:#f85149;
             border-radius:6px; padding:5px 11px; font-weight:700;
             font-size:0.8rem; }
.spike-off { background:#161b22; border:1px solid #30363d; color:#484f58;
             border-radius:6px; padding:5px 11px; font-size:0.8rem; }

#MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Constants & helpers
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {
    "PPO Agent": ("models/final_ppo.zip",  "models/vecnormalize_ppo.pkl", "ppo"),
    "DQN Agent": ("models/final_dqn.zip",  "models/vecnormalize_dqn.pkl", "dqn"),
    "Baseline":  (None,                     None,                          "baseline"),
}

N_MAX    = 10
Q_MAX    = 500
HIST_LEN = 200
GAME_LEN = 200

ACTION_NAMES = {0: "Scale Out (+1)", 1: "Hold (0)", 2: "Scale In (-1)"}
ACTION_CSS   = {0: "action-out", 1: "action-hold", 2: "action-in"}


@st.cache_resource(show_spinner="Loading model…")
def load_agent(model_name: str):
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from cloud_env import CloudScalingEnv
    from baseline_agent import RuleBasedBaseline

    zip_path, pkl_path, kind = MODELS[model_name]
    base_env = DummyVecEnv([lambda: CloudScalingEnv()])

    if kind == "baseline":
        venv = VecNormalize(base_env, norm_obs=True, norm_reward=False,
                            clip_obs=5.0, gamma=0.99, training=False)
        return RuleBasedBaseline(), venv, "baseline"

    venv = VecNormalize.load(pkl_path, base_env)
    venv.training    = False
    venv.norm_reward = False
    model = (PPO if kind == "ppo" else DQN).load(zip_path, env=venv)
    return model, venv, kind


def get_raw_env(venv):
    return venv.venv.envs[0]


def inject_spike(venv):
    raw = get_raw_env(venv)
    if hasattr(raw, "traffic") and raw.traffic is not None:
        raw.traffic._spike_remaining = 15


def is_spiking(venv):
    raw = get_raw_env(venv)
    if hasattr(raw, "traffic") and raw.traffic is not None:
        return raw.traffic._spike_remaining > 0
    return False


def reset_sim(model_name: str):
    model, venv, kind = load_agent(model_name)
    obs = venv.reset()
    raw = get_raw_env(venv)
    ss  = st.session_state
    ss.sim_model   = model
    ss.sim_venv    = venv
    ss.sim_kind    = kind
    ss.sim_obs     = obs
    ss.sim_step    = 0
    ss.sim_reward  = 0.0
    ss.sim_dropped = 0
    ss.sim_cost    = 0
    ss.sim_last_action = 1
    ss.sim_last_reward = 0.0
    ss.sim_running = False
    ss.sim_done    = False
    ss.sim_hist = {k: [] for k in
                   ["t","lam","capacity","queue","active","booting","reward"]}


def sim_step_once():
    ss  = st.session_state
    raw = get_raw_env(ss.sim_venv)

    if ss.sim_kind == "baseline":
        a, _ = ss.sim_model.predict(ss.sim_obs[0], deterministic=True)
        action = np.array([a])
    else:
        action, _ = ss.sim_model.predict(ss.sim_obs, deterministic=True)

    new_obs, rew, dones, infos = ss.sim_venv.step(action)
    info = infos[0]
    raw_r = float(rew[0])

    ss.sim_reward  += raw_r
    ss.sim_dropped += int(info.get("dropped", 0))
    ss.sim_cost    += int(info.get("active",  0))
    ss.sim_last_reward = raw_r
    ss.sim_last_action = int(action.flat[0])
    ss.sim_obs     = new_obs
    ss.sim_step   += 1

    h = ss.sim_hist
    h["t"].append(ss.sim_step)
    h["lam"].append(float(info.get("lambda", 0)))
    h["capacity"].append(int(info.get("active", 0)) * 50)
    h["queue"].append(int(info.get("queue", 0)))
    h["active"].append(int(info.get("active", 0)))
    h["booting"].append(len(raw.boot_timers))
    h["reward"].append(raw_r)
    for k in h:
        if len(h[k]) > HIST_LEN:
            h[k].pop(0)

    if dones[0] or ss.sim_step >= 1000:
        ss.sim_running = False
        ss.sim_done    = True


# ── Game helpers ──────────────────────────────────────────────────────────────

def reset_game(agent_name: str, difficulty: str):
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from cloud_env import CloudScalingEnv

    diff_map = {"Easy": 0.02, "Medium": 0.05, "Hard": 0.10}
    spike_p  = diff_map[difficulty]

    # ── Human env — raw, seed=42 ──────────────────────────────────────────────
    h_env = CloudScalingEnv(seed=42)
    h_env.reset(seed=42)
    h_env.traffic.p_spike = spike_p

    # ── Agent env — wrapped, same seed ────────────────────────────────────────
    a_kind = MODELS[agent_name][2]
    a_base = DummyVecEnv([lambda: CloudScalingEnv(seed=42)])
    if a_kind == "baseline":
        from baseline_agent import RuleBasedBaseline
        a_venv = VecNormalize(a_base, norm_obs=True, norm_reward=False,
                              clip_obs=5.0, gamma=0.99, training=False)
        a_venv.reset()
        get_raw_env(a_venv).traffic.p_spike = spike_p
        a_model = RuleBasedBaseline()
    else:
        zip_path, pkl_path, _ = MODELS[agent_name]
        a_venv = VecNormalize.load(pkl_path, a_base)
        a_venv.training    = False
        a_venv.norm_reward = False
        a_model = (PPO if a_kind == "ppo" else DQN).load(zip_path, env=a_venv)
        a_venv.reset()
        get_raw_env(a_venv).traffic.p_spike = spike_p

    ss = st.session_state
    ss.game_h_env       = h_env
    ss.game_a_model     = a_model
    ss.game_a_venv      = a_venv
    ss.game_a_kind      = a_kind
    ss.game_a_obs       = a_venv.reset()
    # Set spike probability again after the final obs-capturing reset
    get_raw_env(a_venv).traffic.p_spike = spike_p

    ss.game_step        = 0
    ss.game_score_h     = 0.0
    ss.game_score_a     = 0.0
    ss.game_dropped_h   = 0
    ss.game_dropped_a   = 0
    ss.game_last_h      = 1
    ss.game_last_a      = 1
    ss.game_reward_h    = 0.0
    ss.game_reward_a    = 0.0
    ss.game_done        = False
    ss.game_coach       = "Game started. Make your first scaling decision."
    ss.game_waiting     = True
    ss.game_lam_prev    = 45.0
    ss.game_difficulty  = difficulty
    ss.game_agent_name  = agent_name
    ss.game_hist_h      = []
    ss.game_hist_a      = []
    ss.game_hist_lam    = []
    ss.game_hist_t      = []


def coaching_msg(h_action, a_action, lam_now, lam_prev, queue_h, queue_a,
                 active_h, active_a):
    trend   = lam_now - lam_prev
    h_name  = ACTION_NAMES[h_action]
    a_name  = ACTION_NAMES[a_action]

    if h_action == a_action:
        return (f"Both chose {h_name}. "
                f"Demand rate: {lam_now:.1f} req/step. "
                f"Your queue: {queue_h}  |  Agent queue: {queue_a}.")

    lines = [f"You chose {h_name}. Agent chose {a_name}."]

    if a_action == 0:
        if trend > 4:
            lines.append(
                f"Traffic rose by {trend:.1f} in the last step. "
                f"With boot_delay=3, the agent pre-warmed now to absorb demand before the queue grows.")
        elif queue_a > 60:
            lines.append(
                f"Queue reached {queue_a} — agent added capacity before the latency penalty compounds.")
        else:
            lines.append(
                f"The agent sensed rising demand and scaled out early. "
                f"Watch whether this pays off in 3 steps.")

    elif a_action == 2:
        if lam_now < 30 and queue_a == 0:
            lines.append(
                f"Traffic is in a quiet phase (lambda={lam_now:.1f}) and the queue is empty. "
                f"Every idle server costs 1 unit/step with no SLA benefit.")
        else:
            lines.append(
                f"The agent trimmed one server. It expects lambda to stay low enough "
                f"for the remaining {active_a-1} servers.")

    elif a_action == 1:
        if h_action == 0:
            lines.append(
                f"Scaling out costs 1 extra server-timestep AND locks capacity in for 3 boot steps. "
                f"Agent judged current capacity sufficient at lambda={lam_now:.1f}.")
        else:
            lines.append(
                f"Agent held. Current capacity vs demand ratio is manageable. "
                f"Scaling in when spikes are possible would expose the cluster to drop penalties.")

    return " ".join(lines)


def game_step(human_action: int):
    ss  = st.session_state
    raw_h = ss.game_h_env

    # ── human env step ────────────────────────────────────────────────────────
    obs_h, r_h, term_h, trunc_h, info_h = raw_h.step(human_action)
    r_h = float(r_h)

    # ── agent env step ────────────────────────────────────────────────────────
    a_kind = ss.game_a_kind
    if a_kind == "baseline":
        a_arr, _ = ss.game_a_model.predict(ss.game_a_obs[0], deterministic=True)
        a_action = int(a_arr.flat[0])
    else:
        a_arr, _ = ss.game_a_model.predict(ss.game_a_obs, deterministic=True)
        a_action = int(a_arr.flat[0])

    new_a_obs, r_a, _, infos_a = ss.game_a_venv.step(a_arr)
    r_a   = float(r_a[0])
    info_a = infos_a[0]

    # ── update scores ─────────────────────────────────────────────────────────
    ss.game_score_h   += r_h
    ss.game_score_a   += r_a
    ss.game_dropped_h += int(info_h.get("dropped", 0))
    ss.game_dropped_a += int(info_a.get("dropped", 0))
    ss.game_last_h    = human_action
    ss.game_last_a    = a_action
    ss.game_reward_h  = r_h
    ss.game_reward_a  = r_a
    ss.game_a_obs     = new_a_obs
    ss.game_step     += 1

    # ── coaching ──────────────────────────────────────────────────────────────
    lam_now = float(info_h.get("lambda", ss.game_lam_prev))
    ss.game_coach = coaching_msg(
        human_action, a_action, lam_now, ss.game_lam_prev,
        int(info_h.get("queue", 0)), int(info_a.get("queue", 0)),
        raw_h.active, get_raw_env(ss.game_a_venv).active,
    )
    ss.game_lam_prev = lam_now

    # ── history ───────────────────────────────────────────────────────────────
    ss.game_hist_t.append(ss.game_step)
    ss.game_hist_lam.append(lam_now)
    ss.game_hist_h.append(int(info_h.get("queue", 0)))
    ss.game_hist_a.append(int(info_a.get("queue", 0)))

    if trunc_h or ss.game_step >= GAME_LEN:
        ss.game_done    = True
        ss.game_waiting = False
    else:
        ss.game_waiting = True


def rack_html(active, booting, n_max=N_MAX):
    html = '<div class="rack-row">'
    for i in range(n_max):
        if i < active:
            html += '<span class="srv-active">S</span>'
        elif i < active + booting:
            html += '<span class="srv-booting">B</span>'
        else:
            html += '<span class="srv-empty">-</span>'
    html += '</div>'
    return html


# ── Session state bootstrap ───────────────────────────────────────────────────
if "sim_obs" not in st.session_state:
    reset_sim("PPO Agent")

if "game_step" not in st.session_state:
    reset_game("PPO Agent", "Medium")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### RL-Ops Command Center")
    st.caption("CISC 856 · Team 10 · Queen's University")
    st.divider()

    st.markdown("""
    <div style="font-size:0.72rem; color:#8b949e; line-height:1.8;">
    Mahmoud Alyosify &nbsp;·&nbsp; Environment Architect<br>
    Mohamed Yahya &nbsp;·&nbsp; PPO Lead<br>
    Sherouk Rashad &nbsp;·&nbsp; DQN & Sparse Updates<br>
    Salma Hamed &nbsp;·&nbsp; Baseline & Evaluation
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    active_tab = st.radio(
        "View",
        ["Live Simulation", "Challenge Mode"],
        label_visibility="collapsed",
    )

    st.divider()

    if active_tab == "Live Simulation":
        st.markdown('<div class="sec-header">Agent</div>', unsafe_allow_html=True)
        chosen_model = st.selectbox("Model", list(MODELS.keys()),
                                    label_visibility="collapsed")
        if chosen_model != st.session_state.get("_last_model"):
            st.session_state["_last_model"] = chosen_model
            reset_sim(chosen_model)
            st.rerun()

        st.divider()
        st.markdown('<div class="sec-header">Speed</div>', unsafe_allow_html=True)
        speed = st.slider("Steps / second", 1, 20, 5, label_visibility="collapsed")
        delay = 1.0 / speed

        st.divider()
        st.markdown('<div class="sec-header">Traffic</div>', unsafe_allow_html=True)
        spike_btn = "Spike Active" if is_spiking(st.session_state.sim_venv) \
                    else "Inject Traffic Spike"
        if st.button(spike_btn, use_container_width=True, type="primary"):
            inject_spike(st.session_state.sim_venv)
            st.rerun()

        st.divider()
        st.markdown('<div class="sec-header">Controls</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.session_state.sim_running:
                if st.button("Pause", use_container_width=True):
                    st.session_state.sim_running = False
                    st.rerun()
            else:
                if st.button("Run", use_container_width=True, type="primary"):
                    st.session_state.sim_running = True
                    st.session_state.sim_done    = False
                    st.rerun()
        with c2:
            if st.button("Step", use_container_width=True):
                sim_step_once(); st.rerun()

        if st.button("Reset", use_container_width=True):
            reset_sim(chosen_model); st.rerun()

        st.divider()
        ts = st.session_state.sim_step
        st.progress(ts / 1000, text=f"Step {ts} / 1000")
        if st.session_state.sim_done:
            st.success("Episode complete.")

    else:  # Challenge Mode sidebar
        st.markdown('<div class="sec-header">Game Setup</div>',
                    unsafe_allow_html=True)
        game_agent = st.selectbox("Opponent", list(MODELS.keys()),
                                  label_visibility="collapsed")
        difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"],
                                  index=1, label_visibility="collapsed")
        if st.button("New Game", use_container_width=True, type="primary"):
            reset_game(game_agent, difficulty); st.rerun()

        st.divider()
        ss_g = st.session_state
        st.progress(min(1.0, ss_g.game_step / GAME_LEN),
                    text=f"Step {ss_g.game_step} / {GAME_LEN}")
        if ss_g.game_done:
            if ss_g.game_score_h >= ss_g.game_score_a:
                st.success("You win!")
            else:
                st.warning("Agent wins.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB: LIVE SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
if active_tab == "Live Simulation":
    ss  = st.session_state
    raw = get_raw_env(ss.sim_venv)

    # Header
    spike_html = (
        '<span class="spike-on">SPIKE ACTIVE</span>'
        if is_spiking(ss.sim_venv)
        else '<span class="spike-off">No Spike</span>'
    )
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
                border-bottom:1px solid #21262d;padding-bottom:10px;
                margin-bottom:14px;">
      <div>
        <span style="font-size:1.3rem;font-weight:700;">
          RL-Ops Cloud Command Center</span>
        <span style="color:#8b949e;margin-left:12px;font-size:0.82rem;">
          Policy: <b style="color:#58a6ff;">
          {st.session_state.get('_last_model','PPO Agent')}</b>
        </span>
      </div>
      {spike_html}
    </div>""", unsafe_allow_html=True)

    # KPIs
    k1,k2,k3,k4,k5 = st.columns(5)
    active  = raw.active
    booting = len(raw.boot_timers)
    queue   = raw.queue
    cpu_pct = min(100, int((queue + raw.arrival_ema) / max(1, active*50)*100))
    lam     = raw.arrival_ema

    def kpi(col, label, val, css=""):
        col.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value {css}">{val}</div></div>',
            unsafe_allow_html=True)

    kpi(k1, "Cumulative Cost",    f"{ss.sim_cost:,}")
    kpi(k2, "Dropped Requests",   f"{ss.sim_dropped:,}",
        "kpi-danger" if ss.sim_dropped > 0 else "kpi-good")
    kpi(k3, "CPU Utilisation",    f"{cpu_pct}%",
        "kpi-danger" if cpu_pct>85 else "kpi-warn" if cpu_pct>65 else "kpi-good")
    kpi(k4, "Queue Depth",        str(queue),
        "kpi-danger" if queue>300 else "kpi-warn" if queue>100 else "kpi-good")
    kpi(k5, "Arrival Rate",       f"{lam:.1f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    lc, rc = st.columns([3,2])
    with lc:
        st.markdown('<div class="sec-header">Supply vs. Demand</div>',
                    unsafe_allow_html=True)
        if ss.sim_hist["t"]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ss.sim_hist["t"], y=ss.sim_hist["lam"],
                name="Arrival Rate", fill="tozeroy",
                line=dict(color="#d29922",width=1.5),
                fillcolor="rgba(210,153,34,.12)"))
            fig.add_trace(go.Scatter(
                x=ss.sim_hist["t"], y=ss.sim_hist["capacity"],
                name="Provisioned Capacity",
                line=dict(color="#3fb950",width=2)))
            fig.update_layout(
                height=210, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font=dict(color="#8b949e",size=10),
                margin=dict(l=40,r=8,t=8,b=28),
                legend=dict(orientation="h",y=1.1,bgcolor="rgba(0,0,0,0)"),
                xaxis=dict(gridcolor="#21262d"),
                yaxis=dict(gridcolor="#21262d"))
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar":False})

    with rc:
        st.markdown('<div class="sec-header">Queue & Latency</div>',
                    unsafe_allow_html=True)
        if ss.sim_hist["t"]:
            fig2 = go.Figure()
            fig2.add_hrect(y0=300,y1=500,fillcolor="rgba(248,81,73,.07)",
                           line_width=0)
            fig2.add_hrect(y0=100,y1=300,fillcolor="rgba(210,153,34,.07)",
                           line_width=0)
            fig2.add_trace(go.Scatter(
                x=ss.sim_hist["t"], y=ss.sim_hist["queue"],
                fill="tozeroy", line=dict(color="#58a6ff",width=2),
                fillcolor="rgba(88,166,255,.10)", showlegend=False))
            fig2.update_layout(
                height=210, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font=dict(color="#8b949e",size=10),
                margin=dict(l=40,r=8,t=8,b=28), showlegend=False,
                xaxis=dict(gridcolor="#21262d"),
                yaxis=dict(gridcolor="#21262d",range=[0,Q_MAX+20]))
            st.plotly_chart(fig2, use_container_width=True,
                            config={"displayModeBar":False})

    # Datacenter rack
    st.markdown('<div class="sec-header">Datacenter</div>',
                unsafe_allow_html=True)
    dc, ac = st.columns([3,1])
    with dc:
        st.markdown(rack_html(active, booting), unsafe_allow_html=True)
        st.markdown(
            f"<small style='color:#8b949e;'>"
            f"<b style='color:#3fb950;'>{active}</b> active &nbsp;|&nbsp; "
            f"<b style='color:#d29922;'>{booting}</b> booting &nbsp;|&nbsp; "
            f"<b style='color:#484f58;'>{N_MAX-active-booting}</b> idle</small>",
            unsafe_allow_html=True)
    with ac:
        act = ss.sim_last_action
        st.markdown(
            f'<div class="{ACTION_CSS[act]}">{ACTION_NAMES[act]}</div>',
            unsafe_allow_html=True)
        r_color = "#f85149" if ss.sim_last_reward < -50 else "#3fb950"
        st.markdown(
            f"<div style='text-align:center;margin-top:6px;font-size:.82rem;"
            f"color:#8b949e;'>Reward: "
            f"<b style='color:{r_color};'>{ss.sim_last_reward:.2f}</b></div>",
            unsafe_allow_html=True)

    # Cumulative reward
    if ss.sim_hist["t"]:
        cum = np.cumsum(ss.sim_hist["reward"]).tolist()
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=ss.sim_hist["t"], y=cum,
            line=dict(color="#58a6ff",width=1.5),
            fill="tozeroy", fillcolor="rgba(88,166,255,.08)",
            showlegend=False))
        fig3.update_layout(
            height=80, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            margin=dict(l=60,r=8,t=4,b=20),
            font=dict(color="#8b949e",size=9),
            xaxis=dict(gridcolor="#21262d",showgrid=False),
            yaxis=dict(gridcolor="#21262d",title="Cumul. R"),
            annotations=[dict(x=.01,y=.85,xref="paper",yref="paper",
                text=f"Cumulative Reward: {ss.sim_reward:,.1f}",
                showarrow=False,font=dict(color="#8b949e",size=10),
                xanchor="left")])
        st.plotly_chart(fig3, use_container_width=True,
                        config={"displayModeBar":False})

    # Auto-run loop
    if ss.sim_running and not ss.sim_done:
        sim_step_once()
        time.sleep(delay)
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB: CHALLENGE MODE
# ─────────────────────────────────────────────────────────────────────────────
else:
    ss = st.session_state
    raw_h  = ss.game_h_env
    raw_a  = get_raw_env(ss.game_a_venv)

    # ── Header / scoreboard ───────────────────────────────────────────────────
    h_lead = ss.game_score_h >= ss.game_score_a
    st.markdown(f"""
    <div style="border-bottom:1px solid #21262d;padding-bottom:10px;
                margin-bottom:16px;display:flex;justify-content:space-between;
                align-items:center;">
      <div>
        <span style="font-size:1.25rem;font-weight:700;">
          Human vs. {ss.game_agent_name}</span>
        <span style="color:#8b949e;font-size:0.8rem;margin-left:12px;">
          {ss.game_difficulty} · Step {ss.game_step}/{GAME_LEN}</span>
      </div>
      <div style="font-size:0.78rem;color:#8b949e;">
        {'You are ahead' if h_lead else 'Agent is ahead'}
      </div>
    </div>""", unsafe_allow_html=True)

    sc_h = int(ss.game_score_h)
    sc_a = int(ss.game_score_a)
    s1, s2, s3 = st.columns([2,1,2])
    with s1:
        st.markdown(f"""
        <div class="game-panel" style="text-align:center;">
          <div class="game-title">You</div>
          <div class="score-human">{sc_h}</div>
          <div class="score-lead">Dropped: {ss.game_dropped_h}</div>
        </div>""", unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
        <div style="text-align:center;padding:28px 0;
                    font-size:1.4rem;font-weight:700;color:#30363d;">
          vs
        </div>""", unsafe_allow_html=True)
    with s3:
        st.markdown(f"""
        <div class="game-panel" style="text-align:center;">
          <div class="game-title">{ss.game_agent_name}</div>
          <div class="score-agent">{sc_a}</div>
          <div class="score-lead">Dropped: {ss.game_dropped_a}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two cluster panels ────────────────────────────────────────────────────
    lc, rc = st.columns(2)

    with lc:
        st.markdown('<div class="sec-header">Your Cluster</div>',
                    unsafe_allow_html=True)
        st.markdown(rack_html(raw_h.active, len(raw_h.boot_timers)),
                    unsafe_allow_html=True)
        cpu_h = min(100, int((raw_h.queue + raw_h.arrival_ema)
                             / max(1, raw_h.active*50)*100))
        st.markdown(
            f"<small style='color:#8b949e;'>"
            f"Active: <b style='color:#3fb950;'>{raw_h.active}</b> &nbsp;"
            f"Boot: <b style='color:#d29922;'>{len(raw_h.boot_timers)}</b> &nbsp;"
            f"Queue: <b style='color:#f85149;"
            f"'>{raw_h.queue}</b> &nbsp;"
            f"CPU: {cpu_h}%</small>",
            unsafe_allow_html=True)

        if not ss.game_done:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Your decision:**")
            b1, b2, b3 = st.columns(3)
            if b1.button("Scale Out", use_container_width=True, type="primary",
                         disabled=not ss.game_waiting):
                game_step(0); st.rerun()
            if b2.button("Hold", use_container_width=True,
                         disabled=not ss.game_waiting):
                game_step(1); st.rerun()
            if b3.button("Scale In", use_container_width=True,
                         disabled=not ss.game_waiting):
                game_step(2); st.rerun()

            # last action badge
            if ss.game_step > 0:
                act_h = ss.game_last_h
                st.markdown(
                    f'<div class="{ACTION_CSS[act_h]}" '
                    f'style="margin-top:10px;">'
                    f'Last: {ACTION_NAMES[act_h]}</div>',
                    unsafe_allow_html=True)

    with rc:
        st.markdown(f'<div class="sec-header">{ss.game_agent_name} Cluster</div>',
                    unsafe_allow_html=True)
        st.markdown(rack_html(raw_a.active, len(raw_a.boot_timers)),
                    unsafe_allow_html=True)
        cpu_a = min(100, int((raw_a.queue + raw_a.arrival_ema)
                             / max(1, raw_a.active*50)*100))
        st.markdown(
            f"<small style='color:#8b949e;'>"
            f"Active: <b style='color:#3fb950;'>{raw_a.active}</b> &nbsp;"
            f"Boot: <b style='color:#d29922;'>{len(raw_a.boot_timers)}</b>"
            f" &nbsp;Queue: <b style='color:#f85149;"
            f"'>{raw_a.queue}</b> &nbsp;"
            f"CPU: {cpu_a}%</small>",
            unsafe_allow_html=True)

        if ss.game_step > 0:
            st.markdown("<br>", unsafe_allow_html=True)
            act_a = ss.game_last_a
            st.markdown(
                f'<div class="{ACTION_CSS[act_a]}">'
                f'Agent: {ACTION_NAMES[act_a]}</div>',
                unsafe_allow_html=True)

    # ── Coaching panel ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="coach-box">
      <div class="coach-label">Coaching Panel</div>
      {ss.game_coach}
    </div>""", unsafe_allow_html=True)

    # ── Live traffic chart ────────────────────────────────────────────────────
    if ss.game_hist_t:
        st.markdown('<div class="sec-header" style="margin-top:16px;">'
                    'Queue Comparison</div>',
                    unsafe_allow_html=True)
        fig_g = go.Figure()
        fig_g.add_trace(go.Scatter(
            x=ss.game_hist_t, y=ss.game_hist_h,
            name="Your Queue", line=dict(color="#58a6ff",width=1.8)))
        fig_g.add_trace(go.Scatter(
            x=ss.game_hist_t, y=ss.game_hist_a,
            name="Agent Queue", line=dict(color="#3fb950",width=1.8,dash="dot")))
        fig_g.add_hrect(y0=100,y1=Q_MAX,fillcolor="rgba(248,81,73,.05)",
                        line_width=0)
        fig_g.update_layout(
            height=160, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#8b949e",size=10),
            margin=dict(l=40,r=8,t=8,b=24),
            legend=dict(orientation="h",y=1.12,bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d",range=[0,Q_MAX+20]))
        st.plotly_chart(fig_g, use_container_width=True,
                        config={"displayModeBar":False})

    # ── End of game summary ───────────────────────────────────────────────────
    if ss.game_done:
        st.divider()
        you_win = ss.game_score_h >= ss.game_score_a
        result_color = "#58a6ff" if you_win else "#3fb950"
        winner_text  = "You win." if you_win else f"{ss.game_agent_name} wins."
        margin       = abs(int(ss.game_score_h - ss.game_score_a))
        st.markdown(f"""
        <div style="text-align:center;padding:24px;background:#161b22;
                    border:1px solid #30363d;border-radius:10px;margin-top:12px;">
          <div style="font-size:1.5rem;font-weight:700;
                      color:{result_color};">{winner_text}</div>
          <div style="color:#8b949e;font-size:0.9rem;margin-top:6px;">
            Margin: {margin} pts over {GAME_LEN} steps<br>
            Your drops: {ss.game_dropped_h} &nbsp;|&nbsp;
            Agent drops: {ss.game_dropped_a}
          </div>
        </div>""", unsafe_allow_html=True)